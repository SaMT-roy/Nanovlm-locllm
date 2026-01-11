from llm_lora import *
import timm
import torch
from tokenizers import Tokenizer
from torchvision.transforms import v2 as T
import torchvision.io as io
import json
import pandas as pd
import numpy as np
import os
import glob
import ast
from PIL import Image
import re
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from typing import List
from torch.optim.lr_scheduler import _LRScheduler
import math
import cv2
import ast
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()

tokenizer = Tokenizer.from_file("/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_nanovlm/nanovlm3/model_weights/tokenizer.json")

IMG_SEQ_LEN = 49

pad_id = tokenizer.encode("<empty_output>").ids
bos_id = tokenizer.encode("<|im_start|>").ids
eos_id = tokenizer.encode("<|im_end|>").ids
sep_id = tokenizer.encode("<file_sep>").ids
img_token_id = tokenizer.encode("<filename>").ids

system_prompt1 = tokenizer.encode(
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
).ids

system_prompt2 = tokenizer.encode(
    "<|im_end|>\n<|im_start|>system\nAnalyze the provided information and answer the following question.<|im_end|>\n<|im_start|>user\n"
).ids

# Encode the assistant prompt and separator
assistant_prompt = tokenizer.encode(
    "<|im_end|>\n<|im_start|>assistant\n"
).ids


df1 = pd.read_json('/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/nanovlm/data/dpo_data/dpo_unified.jsonl', lines=True)

pattern = "/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/multimodal_pixelshuffle/smollm2_135M/gqa_dpo/gqa_dpo_shard*.jsonl"
files = sorted(glob.glob(pattern))
df2 = pd.concat(
    (pd.read_json(f, lines=True) for f in files),
    ignore_index=True
)

df = pd.concat([df1,df2])
print(df.shape,df.columns)
del df1,df2

lens = []

for p, c, r in tqdm(zip(df["prompt"], df['chosen'], df["rejected"]),
                 total=len(df)):
    if len(c)>len(r): o=c
    else: o=r
    
    try:
        iids   = tokenizer.encode(p).ids
        tids   = tokenizer.encode(o).ids
        lens.append(len(iids+tids))
    except:
        print(p,o)

lens = np.array(lens)

# ------------------------------------------------------------
# 2)  Print key stats
# ------------------------------------------------------------
def pct(x): return np.percentile(lens, x)

print(f"Total samples    : {len(lens):,}")
print(f"Min / Max tokens : {lens.min()} / {lens.max()}")
print(f"Mean ± std       : {lens.mean():.1f} ± {lens.std():.1f}")
print("--- percentiles (tokens) ---")
for p in (50, 90, 95, 98, 99):
    print(f"{p:>3}% : {pct(p):.0f}")

prompt = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{IMAGE TOKENS}<|im_end|>\n"
    "<|im_start|>system\n"
    "Analyze the provided information and answer the following question.<|im_end|>\n"
    "<|im_start|>user\n"
    "{TEXT TOKENS}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

MAX_LEN = IMG_SEQ_LEN + int(pct(99)) + len(tokenizer.encode(prompt).ids)

def encode_pair(text_a: str, text_b: str) -> np.ndarray:
    # Add image tokens
    image_tokens = img_token_id * IMG_SEQ_LEN

    # Encode the first user input
    user_input = tokenizer.encode(text_a).ids

    # Encode the second user input
    assistant_response = tokenizer.encode(text_b).ids

    # Combine all parts
    encoded = system_prompt1 + image_tokens + system_prompt2 + user_input + assistant_prompt + sep_id + assistant_response

    # Truncate to maximum length
    return encoded[:MAX_LEN]

def encode_example(text: str, summary: str):
    ids    = encode_pair(text, summary)     
    labels = ids[1:]

    ids = ids + pad_id*(MAX_LEN-len(ids))
    
    labels = labels + eos_id 
    labels = labels + pad_id*(MAX_LEN-len(labels))

    ids    = np.array(ids,dtype=np.int32)
    labels = np.array(labels,dtype=np.int32)

    # find SEP
    SEP_idxs = np.where(labels == sep_id)[0]
    SEP_pos  = int(SEP_idxs[0]) if SEP_idxs.size else len(ids)

    # build base mask: 1 only for positions > sep_pos AND not PAD
    positions = np.arange(len(labels))
    loss_mask = (positions > SEP_pos).astype(np.float32) * (labels != pad_id).astype(np.float32)

    # Remove SEP from ids, labels, and loss_mask
    sep_mask_ids    = (ids != sep_id[0])
    sep_mask_labels = (labels != sep_id[0])
    
    ids       = ids[sep_mask_ids]
    labels    = labels[sep_mask_labels]
    loss_mask = loss_mask[sep_mask_labels]

    return ids, labels.astype(np.int32), loss_mask

class VLMDataset(Dataset):
    def __init__(self, df, tokenizer, size=224, max_samples=None):

        if max_samples:
            df = df.sample(n=max_samples).reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.size = size

        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        s = self.df.iloc[idx]

        img = cv2.imread(s["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H, W = img.shape[:2]
        img = cv2.resize(img, (self.size, self.size))

        img = self.transform(img)

        chosen_ids, chosen_labels, chosen_loss_mask = encode_example(s['prompt'], s['chosen'])
        rejected_ids, rejected_labels, rejected_loss_mask = encode_example(s['prompt'], s['rejected'])

        return {
            "image": img,
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
            "chosen_loss_mask": torch.tensor(chosen_loss_mask, dtype=torch.float32),
            "rejected_loss_mask": torch.tensor(rejected_loss_mask, dtype=torch.float32)
        }
    
dataset = VLMDataset(df, tokenizer)

train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    sampler=train_sampler,
    pin_memory=True,
    drop_last=True,
    shuffle=False
)

del(dataset)
del(df)

class LMConfig:
    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    lm_tie_weights: bool = True
    lm_model_type: str = 'smollm2-135m'
    lm_chat_template: str = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
        "'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )

# Use it in VLM
cfg = LMConfig()

class PixelUnshuffle(torch.nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        """
        Manual implementation of Pixel Unshuffle.
        Input shape: (Batch, Channels, Height, Width) -> (B, C, H, W)
        Output shape: (Batch, Channels * r^2, H/r, W/r)
        """
        b, c, h, w = x.shape
        r = self.downscale_factor
        
        # 1. Split H and W into (H/r, r) and (W/r, r)
        assert h % r == 0 and w % r == 0, \
            f"H and W must be divisible by r={r}, got {h}, {w}"
        
        out_h, out_w = h // r, w // r
        
        # Reshape to (B, C, H/r, r, W/r, r)
        x = x.view(b, c, out_h, r, out_w, r)
        
        # 2. Permute to bring the 'r' dimensions next to the channels
        # Current: (B, C, H/r, r, W/r, r) -> Target: (B, C, r, r, H/r, W/r)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        
        # 3. Flatten the (C, r, r) into a single dimension
        x = x.view(b, c * (r * r), out_h, out_w)
        
        return x

class VLM(torch.nn.Module):
    def __init__(self, cfg, activate_lora=False):
        super().__init__()
        # --- Language Model ---
        self.decoder = LanguageModel.from_pretrained(cfg, activate_lora=activate_lora)
        self.d_model = cfg.lm_hidden_dim
        self.img_token_id = img_token_id[0]

        # --- Image Embedding Model (ViT) ---
        # Using a small ViT model as an example
        self.img_emb_model = timm.create_model('deit3_small_patch16_224_in21ft1k',pretrained=False)

        self.shuffle_factor = 2
        self.pixel_unshuffle = PixelUnshuffle(self.shuffle_factor)
        self.img_feature_dim = 384*(self.shuffle_factor)**2

        # --- Learnable Query Tokens and Cross-Attention ---
        self.img_seq_len = IMG_SEQ_LEN

        # --- Dense Projector ---
        # This now projects the ViT patch embeddings to the LLM's hidden dimension
        self.dense = torch.nn.Sequential(
                    torch.nn.Linear(self.img_feature_dim, 4*self.d_model),  # expand
                    torch.nn.GELU(),                                      # non-linear activation
                    torch.nn.Linear(4*self.d_model, self.d_model)           # project down to LM dim
                    )

        # Freeze pretrained components
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.img_emb_model.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        img_tensors, tokens = inputs
        img_tensors = img_tensors.to(tokens.device)

        pad_mask = tokens != pad_id[0]

        # --- Get ViT Features ---
        vit_tokens = self.img_emb_model.forward_features(img_tensors)[:, 1:, :]  # (B,196,768)

        B, N, D = vit_tokens.shape
        H = W = int(N ** 0.5)

        vit_feat_map = vit_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
        vit_feat_map = self.pixel_unshuffle(vit_feat_map)  # (B, 1536, 7, 7)

        vit_features = vit_feat_map.flatten(2).transpose(1, 2)  # (B, 49, 1536)

        # --- Project ViT features to LLM's dimension ---
        img_emb = self.dense(vit_features)

        # --- Project and Combine Embeddings ---
        text_emb = self.decoder.token_embedding(tokens)
        final_embeddings = text_emb
    
        placeholder_mask = (tokens == self.img_token_id)

        # Sanity check: Ensure the number of placeholders matches the number of image embeddings
        num_placeholders = placeholder_mask.sum(dim=1)
        if not torch.all(num_placeholders == self.img_seq_len):
            raise ValueError(f"The number of image placeholder tokens in the input ({num_placeholders.tolist()}) "
                             f"does not match the expected number of image embeddings ({self.img_seq_len}).")

        # We replace the embeddings at the masked locations.
        # The shape of `final_embeddings[placeholder_mask]` will be (B * img_seq_len, d_model), 
        # The shape of `img_emb` is (B, img_seq_len, d_model), so we reshape it to match.
        final_embeddings[placeholder_mask] = img_emb.reshape(-1, self.d_model)

        logits = self.decoder(final_embeddings, attention_mask=pad_mask)
        return logits

# Build and load VLM
policy_model = VLM(cfg, activate_lora=True).to(device)
state_dict = torch.load(
    '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/multimodal_pixelshuffle/smollm2_135M/vlm_trained_weights_stage2.pt',
    map_location=device
)

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

policy_model.load_state_dict(new_state_dict, strict=True)

for name, param in policy_model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True # make lora only trainable

policy_model = torch.nn.parallel.DistributedDataParallel(
    policy_model,
    device_ids=[local_rank],
    output_device=local_rank,
)


print("Trainable Parameters:")
print("-" * 60)
total_params = 0
for name, param in policy_model.named_parameters():
    if param.requires_grad:
        param_count = param.numel()
        print(f"Parameter: {name:<50} Shape: {str(param.shape):<20} Parameters: {param_count}")
        total_params += param_count
print("-" * 60)
print(f"Total Trainable Parameters: {total_params:,}")


# Build and load VLM
reference_model = VLM(cfg, activate_lora=True).to(device)
state_dict = torch.load(
    '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/multimodal_pixelshuffle/smollm2_135M/vlm_trained_weights_stage2.pt',
    map_location=device
)

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

reference_model.load_state_dict(new_state_dict, strict=True)
reference_model.eval() 

for param in reference_model.parameters():
    param.requires_grad = False  # Explicitly freeze everything in reference


def compute_dpo_loss(
      model_chosen_logprobs,
      model_rejected_logprobs,
      reference_chosen_logprobs,
      reference_rejected_logprobs,
      beta=1.0,lam=50
    ):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """

    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    chosen_logratios = model_chosen_logprobs - reference_chosen_logprobs
    penalty = torch.clamp(-chosen_logratios, min=0)
    logits = logits - (lam * penalty)

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -torch.nn.functional.logsigmoid(beta * logits)

    # Optional values to track progress during training
    chosen_rewards = chosen_logratios.detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # .mean() to average over the samples in the batch
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

def compute_logprobs(logits, labels, selection_mask=None):
    """
    Compute log probabilities.

    Args:
      logits: Tensor of shape (batch_size, num_tokens, vocab_size)
      labels: Tensor of shape (batch_size, num_tokens)
      selection_mask: Tensor for shape (batch_size, num_tokens)

    Returns:
      mean_log_prob: Mean log probability excluding padding tokens.
    """

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:

        # Apply the mask to filter out padding tokens
        selected_log_probs = selected_log_probs * selection_mask
        return selected_log_probs.sum(-1) / selection_mask.sum(-1)
    
    else:
        return selected_log_probs.mean(-1) 
    

def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    # Move batch to device
    batch = {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    
    # --- POLICY MODEL ---
    # 1. Forward pass for CHOSEN
    policy_chosen_logits = policy_model((batch['image'], batch["chosen_input_ids"]))
    policy_chosen_log_probas = compute_logprobs(
        logits=policy_chosen_logits,
        labels=batch["chosen_labels"],
        selection_mask=batch["chosen_loss_mask"]
    )

    # 2. Forward pass for REJECTED
    policy_rejected_logits = policy_model((batch['image'], batch["rejected_input_ids"]))
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_rejected_logits,
        labels=batch["rejected_labels"],
        selection_mask=batch["rejected_loss_mask"]
    )
    
    # --- REFERENCE MODEL (No Grad) ---
    with torch.no_grad():
        # 1. Forward pass for CHOSEN
        ref_chosen_logits = reference_model((batch['image'], batch["chosen_input_ids"]))
        ref_chosen_log_probas = compute_logprobs(
            logits=ref_chosen_logits,
            labels=batch["chosen_labels"],
            selection_mask=batch["chosen_loss_mask"]
        )

        # 2. Forward pass for REJECTED
        ref_rejected_logits = reference_model((batch['image'], batch["rejected_input_ids"]))
        ref_rejected_log_probas = compute_logprobs(
            logits=ref_rejected_logits,
            labels=batch["rejected_labels"],
            selection_mask=batch["rejected_loss_mask"]
        )

    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta
    )
    return loss, chosen_rewards, rejected_rewards

def train_model_dpo_simple(
    policy_model, reference_model, dataloader,
    optimizer, num_epochs, beta,train_sampler,scheduler=None
):

    # Main training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        policy_model.train()  # Set model to training mode

        running_loss = 0.0
        running_chosen_reward = 0.0
        running_rejected_reward = 0.0
        num_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=True, disable=(local_rank != 0))

        for batch in pbar:

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )

            loss.backward()  # Calculate loss gradients
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)

            optimizer.step()  # Update model weights using loss gradients

            if scheduler:
                scheduler.step()

            # Update metrics
            running_loss += loss.item()
            running_chosen_reward += chosen_rewards.item()
            running_rejected_reward += rejected_rewards.item()
            num_steps += 1

            # Update pbar (Only Rank 0 updates)
            if local_rank == 0:
                avg_loss = running_loss / num_steps
                margin = (running_chosen_reward - running_rejected_reward) / num_steps
                avg_chosen_reward = running_chosen_reward / num_steps
                avg_rejected_reward = running_rejected_reward / num_steps
                
                # Get current LR safely
                current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    chosen_reward=f"{avg_chosen_reward:.4f}",
                    rejected_reward=f"{avg_rejected_reward:.4f}",
                    margin=f"{margin:.4f}", # Reward margin is a better metric than 'acc' for DPO
                    lr=f"{current_lr:.6f}"
                )

        # --- SAVING SECTION (CRITICAL FIXES) ---

        # 1. Synchronization: Wait for ALL GPUs to reach this line
        dist.barrier()

        # 2. Rank Check: Only GPU 0 is allowed to write to disk
        if local_rank == 0:
            save_path = f"policy_model_{epoch}.pt"
            
            print("Saving model...")
            
            # 3. Unwrap DDP: Access .module to get the original model
            # This ensures keys are 'dense.weight', not 'module.dense.weight'
            state_dict_to_save = policy_model.module.state_dict()
            
            # OPTIONAL: Filter to save ONLY trainable weights to save space?
            # If you want to save EVERYTHING, just use state_dict_to_save.
            # If you only want the projector (since LLM/ViT are frozen/standard):
            # state_dict_to_save = {k: v for k, v in vlm.module.state_dict().items() if v.requires_grad}
            
            torch.save(state_dict_to_save, save_path)
            print(f"\n✅ Training completed. Weights saved correctly to: {save_path}")

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps_ratio, min_lr, max_lr, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_steps_ratio)
        self.min_lr = min_lr
        self.max_lr = max_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        if current_step < self.warmup_steps:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (current_step / self.warmup_steps)
        else:
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            decay_factor = math.exp(-1.0 * progress)
            lr = self.min_lr + (self.max_lr - self.min_lr) * decay_factor
            lr = max(lr, self.min_lr)
        return [lr for _ in self.base_lrs]
    
EPOCHS = 3
LEARNING_RATE_MIN = 1e-6
LEARNING_RATE_MAX = 5e-6
WARMUP_RATIO = 0.1

# --- Optimizer ---
optimizer = torch.optim.Adam(policy_model.parameters(), lr=LEARNING_RATE_MAX, betas=(0.9, 0.98), eps=1e-9)
criterion = torch.nn.CrossEntropyLoss(reduction='none')  

# --- Calculate total steps and initialize scheduler ---
total_steps = len(dataloader) * EPOCHS
scheduler = CustomLRScheduler(
    optimizer,
    total_steps=total_steps,
    warmup_steps_ratio=WARMUP_RATIO,
    min_lr=LEARNING_RATE_MIN,
    max_lr=LEARNING_RATE_MAX
)

tracking = train_model_dpo_simple(
    policy_model=policy_model,
    reference_model=reference_model,
    dataloader=dataloader,
    train_sampler=train_sampler,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=EPOCHS,
    beta=1
)

dist.destroy_process_group()
