from llm_lora import *
import timm
import torch
from tokenizers import Tokenizer
from torchvision.transforms import v2 as T
import torchvision.io as io
import json
import pandas as pd
import numpy as np
from typing import List
import os
import cv2
import matplotlib.pyplot as plt
import ast
import re
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--shard_id", type=int, required=True)
parser.add_argument("--num_shards", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0")

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())


def shard_df(df, shard_id, num_shards):
    return df.iloc[shard_id::num_shards].reset_index(drop=True)

tokenizer = Tokenizer.from_file("/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/multimodal_pixelshuffle/smollm2_135M/model_weights/tokenizer.json")

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

vocab = tokenizer.get_vocab()           
id_to_token = {idx: tok for tok, idx in vocab.items()}

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

IMG_SEQ_LEN = 196
    
class VLM(torch.nn.Module):
    def __init__(self, cfg, activate_lora=False):
        super().__init__()
        # --- Language Model ---
        self.decoder = LanguageModel.from_pretrained(cfg, activate_lora=activate_lora)
        self.d_model = cfg.lm_hidden_dim
        self.img_token_id = img_token_id[0]

        # --- Image Embedding Model (ViT) ---
        # Using a small ViT model as an example
        self.img_emb_model = timm.create_model('deit3_base_patch16_224_in21ft1k',pretrained=False)

        self.img_feature_dim = 768

        # --- Learnable Query Tokens and Cross-Attention ---
        self.img_seq_len = IMG_SEQ_LEN

        # --- Dense Projector ---
        # This now projects the ViT patch embeddings to the LLM's hidden dimension
        self.dense = torch.nn.Sequential(
                    torch.nn.Linear(self.img_feature_dim, 4*self.d_model),  # expand
                    torch.nn.GELU(),                                      # non-linear activation
                    torch.nn.Linear(4*self.d_model, self.d_model)           # project down to LM dim
                    )

# Build and load VLM
vlm = VLM(cfg, activate_lora=True)
state_dict = torch.load(
    '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/multimodal/vlm_trained_weights_stage2.pt',
    map_location='cpu'
)

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

vlm.load_state_dict(new_state_dict, strict=True)
vlm.to(device)
vlm.eval()


def encode(text: str) -> List[int]:
    # Add image tokens
    image_tokens = img_token_id * IMG_SEQ_LEN

    # Encode the first user input9
    user_input = tokenizer.encode(text).ids

    full_ids = system_prompt1 + image_tokens + system_prompt2 + user_input + assistant_prompt
    return full_ids

def load_and_preprocess_image(path, transform, device):
    imgbgr = cv2.imread(path)
    imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(imgrgb, (224, 224))
    return transform(resized)


def generate(vlm, img_paths, question, eos_id, pad_id, max_tokens=30):

    mean=(0.485,0.456,0.406)
    std =(0.229,0.224,0.225)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    img_tensors = []
    for p in img_paths:
        img_tensors.append(load_and_preprocess_image(p, transform, device))
    img_tensors = torch.stack(img_tensors).to(device)

    prompt_ids = [encode(q) for q in question]
    max_len = max(len(p) for p in prompt_ids)
    padded_prompt_ids = [
        pad_id* (max_len - len(p)) + p
        for p in prompt_ids
    ]
    text_tensor = torch.tensor(padded_prompt_ids, dtype=torch.int32).to(device)

    # --- Get ViT Features ---
    vit_features = vlm.img_emb_model.forward_features(img_tensors)[:, 1:, :]  # (B,196,768)

    # project to llm space
    img_emb = vlm.dense(vit_features)
    B,_,_ = img_emb.shape

    # --- Prepare for Generation ---
    text_emb = vlm.decoder.token_embedding(text_tensor)
    final_embeddings = text_emb

    placeholder_mask = (text_tensor == img_token_id[0])

    num_placeholders = placeholder_mask.sum(dim=1)
    if not torch.all(num_placeholders == IMG_SEQ_LEN):
        raise ValueError(f"The number of image placeholder tokens in the input ({num_placeholders.tolist()}) "
                            f"does not match the expected number of image embeddings ({IMG_SEQ_LEN}).")

    for b in range(B):
        final_embeddings[b, placeholder_mask[b]] = img_emb[b]

    pad_mask = (text_tensor == pad_id[0])
    final_embeddings[pad_mask] = 0

    generated_outputs = final_embeddings
    finished = torch.zeros(B, dtype=torch.bool, device=generated_outputs.device)
    newly_generated_ids = []

    for step in range(max_tokens):

        # Forward
        prompt_output = vlm.decoder(
            generated_outputs,
            attention_mask=None
        )

        logits = prompt_output[:, -1, :] 
        next_token = torch.argmax(logits, dim=-1)  # (B,)

        # Force EOS for finished sequences
        next_token = torch.where(
            finished,
            torch.full_like(next_token, eos_id[0]),
            next_token
        )

        newly_generated_ids.append(next_token)

        # Update finished mask
        finished |= (next_token == eos_id[0])

        # Embed next token
        next_emb = vlm.decoder.token_embedding(next_token).unsqueeze(1)

        # Append
        generated_outputs = torch.cat([generated_outputs, next_emb], dim=1)

        # Stop when ALL sequences finished
        if finished.all():
            break


    # (T_gen, B)
    gen_ids = torch.stack(newly_generated_ids, dim=0)

    # (B, T_gen)
    gen_ids = gen_ids.transpose(0, 1)

    decoded = []

    for b in range(gen_ids.size(0)):
        seq = gen_ids[b].tolist()
        if eos_id in seq:
            seq = seq[:seq.index(eos_id)]
        decoded.append(tokenizer.decode(seq).strip())

    return decoded


batch_size = 8

json_path = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/nanovlm/data/gqa/train_qa.jsonl'
output_path = f"/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/multimodal/gqa_dpo/gqa_dpo_shard{args.shard_id}.jsonl"

# -----------------------------
# Load data
# -----------------------------
data = []
with open(json_path, "r") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df1 = (
    df
    .groupby('img_path', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), 7), random_state=42))
    .reset_index(drop=True)
)
df1 = shard_df(df1, args.shard_id, args.num_shards)
print(df1.shape)
df1["rejected"] = None

# -----------------------------
# Resume support
# -----------------------------
processed = set()
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        for line in f:
            d = json.loads(line)
            processed.add(d["img_path"] + "||" + d["prompt"])

# -----------------------------
# Main loop
# -----------------------------
for i in tqdm(range(0, len(df1), batch_size)):
    batch_df = df1.iloc[i : i + batch_size]

    # Skip already processed rows
    batch_df = batch_df[
        ~batch_df.apply(
            lambda r: r["img_path"] + "||" + r["prompt"] in processed,
            axis=1
        )
    ]

    if batch_df.empty:
        continue

    outs = generate(
        vlm,
        batch_df["img_path"].tolist(),
        batch_df["prompt"].tolist(),
        eos_id,
        pad_id,
        max_tokens=30,
    )

    batch_df = batch_df.copy()
    batch_df["rejected"] = outs


    with open(output_path, "a") as f:
        for _, row in batch_df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            processed.add(row["img_path"] + "||" + row["prompt"])
