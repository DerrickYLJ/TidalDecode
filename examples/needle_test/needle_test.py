# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import os
from dataclasses import dataclass
from datetime import datetime

from needle_tools import LLMNeedleHaystackTester
from needle_viz import plot_needle_viz

from examples.needle_test.index_example import warm_up
from mpi4py import MPI


@dataclass
class Config:
    # wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl
    haystack_file: str = "data/pg19_mini.jsonl"  # Path to the haystack file
    model_name: str = "01-ai/Yi-9B-200K"
    run_name: str = None  # Name of the run, used for the output file
    context_lengths_min: int = 30_000
    context_lengths_max: int = 100_000
    n_context_length_intervals: int = 15  # Number of intervals between min and max
    n_document_depth_intervals: int = 10  # position of the needle in the haystack
    n_rounds: int = 3
    seed: int = 42
    attn_type: str = "vllm"
    output_path: str = "results/needle/"
    pattern_path: str = "config/Llama_3_8B_Instruct_262k_kv_out_v32_best_pattern.json"
    jobs: str = None
    kv_cache_cpu: bool = False
    trust_remote_code: bool = False
    kv_cache_cpu_device: str = "cpu"
    top_k: int = None
    comm: MPI.Comm = None  # MPI communicator
    sparse_layer_start: int =2
    correction_layer: int =9

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        output_file = f"needle_res_{self.model_name.replace('/', '-')}_{self.run_name if self.run_name is not None else ''}_{self.jobs if self.jobs is not None else ''}_{timestamp}_{self.context_lengths_min}_{self.context_lengths_max}_{self.pattern_path.split('/')[-1].replace('.json', '') if self.pattern_path is not None else ''}.json"
        self.output_file = os.path.join(self.output_path, output_file)


def main(
    model_name: str,
    run_name: str = None,
    attn_type: str = "vllm",
    output_path: str = "results/needle/",
    pattern_path: str = "config/Llama_3_8B_Instruct_262k_kv_out_v32_best_pattern.json",
    rounds: int = 3,
    jobs: str = None,
    max_length: int = 100000,
    min_length: int = 1000,
    kv_cache_cpu: bool = False,
    trust_remote_code: bool = False,
    kv_cache_cpu_device: str = "cpu",
    top_k: int = None,
    sparse_layer_start=2,
    correction_layer=9,
    use_mpi: bool = False,
):
    print(f"top_k: {top_k}; sparse_layer_start: {sparse_layer_start}; correction_layer: {correction_layer}")
    comm = MPI.COMM_WORLD if use_mpi else None
    config = Config(
        model_name=model_name,
        run_name=run_name,
        attn_type=attn_type,
        output_path=output_path,
        pattern_path=pattern_path,
        n_rounds=rounds,
        jobs=jobs,
        context_lengths_min=min_length,
        context_lengths_max=max_length,
        kv_cache_cpu=kv_cache_cpu,
        trust_remote_code=trust_remote_code,
        kv_cache_cpu_device=kv_cache_cpu_device,
        top_k=top_k,
        sparse_layer_start=sparse_layer_start,
        correction_layer=correction_layer,
        comm=comm,
    )
    kwargs = {
        "swap_space": 64,
        "gpu_memory_utilization": 0.98,
    }
    ht = LLMNeedleHaystackTester(config, **kwargs if config.attn_type == "vllm" else {})
    if not use_mpi or comm.Get_rank() == 0:
        ht.start_test()

        print("making plot...")
        plot_needle_viz(
            config.output_file,
            (
                config.model_name.replace("/", "-") + f"_{config.run_name}"
                if config.run_name is not None
                else ""
            ),
            config.context_lengths_min,
            config.context_lengths_max,
            mode=attn_type,
            output_path=config.output_path,
        )


if __name__ == "__main__":
    if True:
        args = argparse.ArgumentParser()
        args.add_argument("--model_name", type=str, default="lmsys/vicuna-13b-v1.3")
        args.add_argument("--run_name", type=str, default=None)
        args.add_argument(
            "--attn_type",
                type=str,
                choices=[
                    "index",
                    "quest"
                ],
                default="index",
        )

        args.add_argument("--top_k", type=int, default=None)
        args.add_argument("--sparse_layer_start", type=int, default=2)
        args.add_argument("--correction_layer", type=int, default=13)
        args.add_argument("--output_path", type=str, default="results/needle/")
        args.add_argument("--pattern_path", type=str, default=None)
        args.add_argument("--rounds", type=int, default=3)
        args.add_argument("--jobs", type=str, default=None)
        args.add_argument("--max_length", type=int, default=100000)
        args.add_argument("--min_length", type=int, default=1000)
        args.add_argument("--kv_cache_cpu", action="store_true")
        args.add_argument("--kv_cache_cpu_device", type=str, default="cpu")
        args.add_argument("--trust_remote_code", action="store_true")
        args.add_argument("--use_mpi", action="store_true")
        args = args.parse_args()
        args.output_path = os.path.join(
            args.output_path,
            str(args.top_k),
            f"{args.min_length//1000}K_{args.max_length//1000}K",
        )


        # warm up program
        # warm_up(2, nq_list, nb_list, k_list)
        main(
            model_name=args.model_name,
            run_name=args.run_name,
            attn_type=args.attn_type,
            output_path=args.output_path,
            pattern_path=args.pattern_path,
            rounds=args.rounds,
            jobs=args.jobs,
            max_length=args.max_length,
            min_length=args.min_length,
            kv_cache_cpu=args.kv_cache_cpu,
            trust_remote_code=args.trust_remote_code,
            kv_cache_cpu_device=args.kv_cache_cpu_device,
            top_k=args.top_k,
            sparse_layer_start=args.sparse_layer_start,
            correction_layer=args.correction_layer,
            use_mpi=args.use_mpi,
        )
    else:
        # tmp ppl test
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
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            LlamaForCausalLM,
        )

        from src.utils import load, download_url, load_jsonl
        from src.enabling_index import enable_src

        import torch
        from tqdm import tqdm
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
        from torch.nn import CrossEntropyLoss

        import argparse
        from argparse import ArgumentParser

        device = "cuda"

        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name_or_path", type=str)
        parser.add_argument("--fixed-length", type=int)
        parser.add_argument("--max-tokens", type=int, default=8192)
        parser.add_argument("--min-tokens", type=int, default=256)
        parser.add_argument("--tokens-step", type=int)
        parser.add_argument("--length-step", type=int, default=128)
        parser.add_argument("--iterations", type=int, default=20)
        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--sparse_layer_start", type=int, default=2)
        parser.add_argument("--correction_layer", type=int, default=9)

        parser.add_argument("--num_eval_tokens", type=int, default=None)

        parser.add_argument(
                "--attn_type",
                type=str,
                choices=[
                    "index",
                    "quest"
                ],
                default="index",
            )
        parser.add_argument("--top_k", type=int, default=1024)
        parser.add_argument("--chunk_size", type=int, default=16)


        def load_model(model_name_or_path, args):
            print(f"Loading model from {model_name_or_path} ...")
            # however, tensor parallel for running falcon will occur bugs

            model, tokenizer = load(model_name_or_path)
            enable_src(model, args.top_k, None, args.attn_type, args.sparse_layer_start, args.correction_layer)

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
