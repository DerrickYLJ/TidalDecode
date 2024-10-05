# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import gc
import json
import os

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import load, download_url, load_jsonl

import torch
from tqdm import tqdm
import os
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

import argparse
from argparse import ArgumentParser

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--sparse_layer_start", type=int, default=2)
parser.add_argument("--correction_layer", type=int, default=9)

parser.add_argument("--num_eval_tokens", type=int, default=None)

parser.add_argument(
    "--attn_type",
    type=str,
    choices=["tidal"],
    default=None,
)
parser.add_argument("--top_k", type=int, default=1024)


def load_model(model_name_or_path, args):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs

    model, tokenizer = load(model_name_or_path, args.attn_type, top_k=args.top_k, sparse_layer_start=args.sparse_layer_start,
        correction_layer=args.correction_layer)
    
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


args = parser.parse_args()

data = load_dataset("emozilla/pg19-test", split="test")

model, tokenizer = load_model(args.model_name_or_path, args)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

num_eval_tokens = 0
for text in data["text"][:1]:
    encodings = tokenizer(text, return_tensors="pt")

    print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)

        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
