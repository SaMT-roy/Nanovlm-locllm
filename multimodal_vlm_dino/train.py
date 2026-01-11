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
import ast
from PIL import Image
import re
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from typing import List
from torch.optim.lr_scheduler import _LRScheduler
import math
from tqdm import tqdm
import cv2
import ast
import matplotlib.pyplot as plt
import torch.distributed as dist
from dinov3_meta import DinoVisionTransformer

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()

tokenizer = Tokenizer.from_file("/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_nanovlm/nanovlm3/model_weights/tokenizer.json")

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

class LMConfig:
    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66  # Number of extra tokens for the VLM (image start, image end, image token)
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount # Not a great way to do this, but it works for now (vlm_extra_tokens cannot be a dict, since this is mutable, and a Field has no len() function)
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    lm_tie_weights: bool = True # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type = 'smollm2-360m'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# Use it in VLM
cfg = LMConfig()

# https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py
def vit_small(**kwargs):
    model = DinoVisionTransformer(
        img_size=256,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        compact_arch_name="vits",
        **kwargs,
    )
    return model

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
    
IMG_SEQ_LEN = 64

class VLM(torch.nn.Module):
    def __init__(self, cfg, activate_lora=False):
        super().__init__()
        # --- Language Model ---
        self.decoder = LanguageModel.from_pretrained(cfg, activate_lora=activate_lora)
        self.d_model = cfg.lm_hidden_dim
        self.img_token_id = img_token_id[0]

        # --- Image Embedding Model (ViT) ---
        # Using a small ViT model as an example
        self.img_emb_model = vit_small()
        state_dict = torch.load("/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/multimodal4_dino_pixelshuffle/dinov3_vits16_pretrain_lvd1689m-08c60483.pth", map_location="cpu")
        self.img_emb_model.load_state_dict(state_dict)

        self.shuffle_factor = 3
        self.pixel_unshuffle = PixelUnshuffle(self.shuffle_factor)
        self.img_feature_dim = 384*(self.shuffle_factor)**2

        # --- Learnable Query Tokens and Cross-Attention ---
        self.img_seq_len = IMG_SEQ_LEN

        # --- Dense Projector ---
        # This now projects the ViT patch embeddings to the LLM's hidden dimension
        self.dense = torch.nn.Sequential(
                    torch.nn.Linear(self.img_feature_dim, self.d_model),  # expand
                    torch.nn.GELU(),                                      # non-linear activation
                    torch.nn.Linear(self.d_model, self.d_model)           # project down to LM dim
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
        vit_tokens = self.img_emb_model.forward_features(img_tensors)['x_norm_patchtokens']

        B, N, D = vit_tokens.shape
        H = W = int(N ** 0.5)

        vit_feat_map = vit_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
        vit_feat_map = self.pixel_unshuffle(vit_feat_map)  # (B, 9*D, 8, 8)

        vit_features = vit_feat_map.flatten(2).transpose(1, 2)  # (B, 64, 3*3*D)

        # --- Project ViT features to LLM's dimension ---
        img_emb = self.dense(vit_features)

        # --- Project and Combine Embeddings ---
        text_emb = self.decoder.token_embedding(tokens)
        final_embeddings = text_emb
    
        placeholder_mask = (tokens == self.img_token_id)
        img_weight_mask = placeholder_mask.unsqueeze(-1).float()

        # Sanity check: Ensure the number of placeholders matches the number of image embeddings
        num_placeholders = placeholder_mask.sum(dim=1)
        if not torch.all(num_placeholders == self.img_seq_len):
            raise ValueError(f"The number of image placeholder tokens in the input ({num_placeholders.tolist()}) "
                             f"does not match the expected number of image embeddings ({self.img_seq_len}).")

        # We replace the embeddings at the masked locations.
        # The shape of `final_embeddings[placeholder_mask]` will be (B * img_seq_len, d_model), 
        # The shape of `img_emb` is (B, img_seq_len, d_model), so we reshape it to match.
        final_embeddings[placeholder_mask] = img_emb.reshape(-1, self.d_model)

        logits = self.decoder(final_embeddings, final_embeddings, attention_mask=pad_mask, img_weight_mask=img_weight_mask)
        return logits
    
    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens=30, temp=1.0):
        img_tensors, tokens = inputs
        img_tensors = img_tensors.to(tokens.device)

        # --- Get ViT Features ---
        vit_tokens = self.img_emb_model.forward_features(img_tensors)['x_norm_patchtokens']

        B, N, D = vit_tokens.shape
        H = W = int(N ** 0.5)

        vit_feat_map = vit_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
        vit_feat_map = self.pixel_unshuffle(vit_feat_map)  # (B, 9*D, 8, 8)

        vit_features = vit_feat_map.flatten(2).transpose(1, 2)  # (B, 64, 3*3*D)
        
        # project to llm space
        img_emb = self.dense(vit_features)

        # --- Prepare for Generation ---
        text_emb = self.decoder.token_embedding(tokens)
        final_embeddings = text_emb

        placeholder_mask = (tokens == self.img_token_id)
        img_weight_mask = placeholder_mask.unsqueeze(-1).float()

        num_placeholders = placeholder_mask.sum(dim=1)
        if not torch.all(num_placeholders == self.img_seq_len):
            raise ValueError(f"The number of image placeholder tokens in the input ({num_placeholders.tolist()}) "
                             f"does not match the expected number of image embeddings ({self.img_seq_len}).")

        final_embeddings[placeholder_mask] = img_emb.reshape(-1, self.d_model)

        # --- Autoregressive Generation Loop ---
        generated_outputs = final_embeddings
        newly_generated_ids_list = []

        for i in range(max_new_tokens):
            # Forward pass
            prompt_output = self.decoder(generated_outputs, generated_outputs, attention_mask=None, img_weight_mask=img_weight_mask)
            last_output = prompt_output[:, -1, :] / temp  

            # --- Strict greedy: take argmax instead of sampling ---
            next_token = torch.argmax(last_output, dim=-1, keepdim=True)

            # Append to sequence
            newly_generated_ids_list.append(next_token)
            next_emb = self.decoder.token_embedding(next_token)
            generated_outputs = torch.cat((generated_outputs, next_emb), dim=1)

            zero_mask = torch.zeros(
                img_weight_mask.size(0), 1, img_weight_mask.size(-1),
                device=img_weight_mask.device,
                dtype=img_weight_mask.dtype
            )
            img_weight_mask = torch.cat((img_weight_mask, zero_mask), dim=1)

            # Check for EOS
            if next_token.item() == 2:  # EOS token ID
                break

        return newly_generated_ids_list


question_pool = [
'',
    'Render a clear and concise summary of the photo.',
    'Write a terse but informative summary of the picture.',
    'Describe the image properly.',
    'Share a proper interpretation of the image provided.',
    'Give a brief description of the image.',
    "Present a compact description of the photo's key features.",
    'Write a terse but informative summary of the picture.',
    'Provide a brief description of the given image.',
    'Summarize the visual content of the image.',
    'What is in the image?',
    'Provide a brief description of the given image.',
    "Present a compact description of the images's key features.",
    'Give a short and clear explanation of the subsequent image.',
    "How would you describe the background setting?",
'',
    "What is happening in the image?",
    "What draws your attention most in the image?",
    "Does the image appear to be candid or posed?",
    'What is unique about the image?',
]

file_path = "/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/nanovlm/data/flickr/flickr30k_train_captions.jsonl"
flickr_image_folder = "/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/nanovlm/data/flickr/flickr30k-images"

rows = []
with open(file_path, "r") as f:
    for line in f:
        item = json.loads(line)
        img_id = str(item['image_id'])      # ensure 12-digit string
        img_file = f"{img_id}.jpg"
        img_path = os.path.join(flickr_image_folder, img_file)
        
        # Randomly pick a question from the pool
        input_text = random.choice(question_pool)
        
        # Caption as target
        target_text = item.get('caption', '')  # fallback to empty string if missing
        
        # Add row
        rows.append({
            'imgpath': img_path,
            'input_text': input_text,
            'target_text': target_text
        })

df1 = pd.DataFrame(rows)
for i in range(10):
    print(df1['imgpath'][i])
del(rows)

file_path = "/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/nanovlm/data/coco/coco_train_captions.jsonl"
coco_image_folder = "/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_train2017"

rows = []
with open(file_path, "r") as f:
    for line in f:
        item = json.loads(line)
        img_id = str(item['image_id']).zfill(12)       # ensure 12-digit string
        img_file = f"{img_id}.jpg"
        img_path = os.path.join(coco_image_folder, img_file)
        
        # Randomly pick a question from the pool
        input_text = random.choice(question_pool)
        
        # Caption as target
        target_text = item.get('caption', '')  # fallback to empty string if missing
        
        # Add row
        rows.append({
            'imgpath': img_path,
            'input_text': input_text,
            'target_text': target_text
        })

df2 = pd.DataFrame(rows)
for i in range(10):
    print(df2['imgpath'][i])
del(rows)

file_path = "/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/nanovlm/data/open_images/open_images_test_captions.jsonl"
open_image_folder = "/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/nanovlm/data/open_images/open_images"

rows = []
with open(file_path, "r") as f:
    for line in f:
        item = json.loads(line)
        img_id = str(item['image_id'])      # ensure 12-digit string
        img_file = f"{img_id}.jpg"
        img_path = os.path.join(open_image_folder, img_file)
        
        # Randomly pick a question from the pool
        input_text = random.choice(question_pool)
        
        # Caption as target
        target_text = item.get('caption', '')  # fallback to empty string if missing
        
        # Add row
        rows.append({
            'imgpath': img_path,
            'input_text': input_text,
            'target_text': target_text
        })

df3 = pd.DataFrame(rows)
for i in range(10):
    print(df3['imgpath'][i])
del(rows)

df = pd.concat([df1,df2,df3]).reset_index(drop=True)

del(df1)
del(df2)
del(df3)

tok = tokenizer          # whatever variable you use

# ------------------------------------------------------------
# 1)  Get token counts for every example
#     (here: concatenate summary + dialogue; split if you want)
# ------------------------------------------------------------
lens = []

for s, d in tqdm(zip(df["input_text"], df["target_text"]),
                 total=len(df)):
    iids   = tok.encode(s).ids
    tids   = tok.encode(d).ids
    
    lens.append(len(iids+tids))

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

        img = cv2.imread(s["imgpath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H, W = img.shape[:2]
        img = cv2.resize(img, (self.size, self.size))

        img = self.transform(img)

        ids, labels, loss_mask = encode_example(s['input_text'], s['target_text'])

        return {
            "image": img,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
        }

print('initial training samples', len(df))
df = df[df['imgpath'].apply(os.path.exists)].reset_index(drop=True)
print("Remaining samples:", len(df))

dataset = VLMDataset(df, tokenizer, size=384)

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

vocab = tokenizer.get_vocab()           
id_to_token = {idx: tok for tok, idx in vocab.items()}

# Build and load VLM
vlm = VLM(cfg).to(device)

vlm = torch.nn.parallel.DistributedDataParallel(
    vlm,
    device_ids=[local_rank],
    output_device=local_rank,
)

if local_rank == 0:

    print("Trainable Parameters:")
    print("-" * 60)
    total_params = 0
    for name, param in vlm.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            print(f"Parameter: {name:<50} Shape: {str(param.shape):<20} Parameters: {param_count}")
            total_params += param_count
    print("-" * 60)
    print(f"Total Trainable Parameters: {total_params:,}")

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
LEARNING_RATE_MIN = 1e-5
LEARNING_RATE_MAX = 1e-3
WARMUP_RATIO = 0.1

# --- Optimizer ---
optimizer = torch.optim.Adam(vlm.parameters(), lr=LEARNING_RATE_MAX, betas=(0.9, 0.98), eps=1e-9)
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

for epoch in range(1, EPOCHS + 1):
    train_sampler.set_epoch(epoch)
    vlm.train()
    tot_loss = tot_correct = tot_tokens = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True, disable=(local_rank != 0))

    for batch in pbar:
        img = batch['image']
        ids = batch["input_ids"].to(device)          # (B, T)
        tgt = batch["labels"].to(device)             # (B, T)
        mask = batch["loss_mask"].to(device)         # (B, T)

        optimizer.zero_grad()
        logits = vlm((img,ids))                          # (B, T, V)

        # ---- loss ----
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))  # (B*T,)
        masked_loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)  # Apply mask and average
        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(vlm.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # ---- metrics (no grad) ----
        with torch.no_grad():
            preds = logits.argmax(-1)                # (B, T)
            valid = (mask == 1.0)                    # bool mask where loss_mask is 1.0
            correct = (preds == tgt) & valid         # Correct predictions where mask is 1.0
            n_tok = valid.sum().item()               # Count of masked tokens

            tot_correct += correct.sum().item()
            tot_tokens += n_tok
            tot_loss += masked_loss.item() * n_tok   # Scale by number of valid tokens

        # Update pbar (Only Rank 0 updates)
        if local_rank == 0:
            pbar.set_postfix(
                loss=f"{tot_loss / tot_tokens:.4f}" if tot_tokens > 0 else "N/A",
                acc=f"{tot_correct / tot_tokens:.4f}" if tot_tokens > 0 else "N/A",
                lr=f"{scheduler.get_last_lr()[0]:.6f}"
            )

    # --- SAVING SECTION (CRITICAL FIXES) ---

    # 1. Synchronization: Wait for ALL GPUs to reach this line
    dist.barrier()

    # 2. Rank Check: Only GPU 0 is allowed to write to disk
    if local_rank == 0:
        save_path = "vlm_trained_weights_stage1.pt"
        
        print("Saving model...")
        
        # 3. Unwrap DDP: Access .module to get the original model
        # This ensures keys are 'dense.weight', not 'module.dense.weight'
        state_dict_to_save = vlm.module.state_dict()
        
        # OPTIONAL: Filter to save ONLY trainable weights to save space?
        # If you want to save EVERYTHING, just use state_dict_to_save.
        # If you only want the projector (since LLM/ViT are frozen/standard):
        # state_dict_to_save = {k: v for k, v in vlm.module.state_dict().items() if v.requires_grad}
        
        torch.save(state_dict_to_save, save_path)
        print(f"\n✅ Training completed. Weights saved correctly to: {save_path}")

# 4. Cleanup
dist.destroy_process_group()
