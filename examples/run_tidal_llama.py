import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from src.utils import load, download_url, load_jsonl


@torch.no_grad()
def tidal_inference(model, tokenizer, prompts, max_gen_len=256):
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_tensor = tokenizer(
            prompt, return_tensors="pt", return_attention_mask=False
        )
        with torch.no_grad():
            outs = model.generate(
                **input_tensor,
                max_new_tokens=max_gen_len,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        new_tokens = outs[0, input_tensor["input_ids"].shape[-1] :]
        out = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(out)


def main(args):
    model_name_or_path = args.model_name
    model, tokenizer = load(model_name_or_path, args.attn_type, top_k=args.top_k, sparse_layer_start=args.sparse_layer_start, correction_layer=args.correction_layer)
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    tidal_inference(
        model,
        tokenizer,
        prompts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    )
    parser.add_argument(
            "--attn_type",
                type=str,
                choices=[
                    "tidal",
                ],
                default=None,
        )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--sparse_layer_start", type=int, default=2)
    parser.add_argument("--correction_layer", type=int, default=13)

    args = parser.parse_args()

    main(args)