# Based on Punica Project
# Check: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/benchmarks/bench_textgen.py

import argparse
import dataclasses
import time
import numpy as np
import torch
from tqdm.auto import tqdm

from tidal import LlamaForCausalLM as td_model
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM as hf_model


@dataclasses.dataclass
class ModelConfig:
    model_path: str
    dtype: str = dataclasses.field(default="float16")
    device: str = dataclasses.field(default="cuda:0")


MODEL_CFGS = {
    "llama2-7b": ModelConfig(model_path="meta-llama/Llama-2-7b-chat-hf"),
    "llama3.1-8b": ModelConfig(model_path="meta-llama/Llama-3.1-8B"),
    "llama3.1-8b-instruct": ModelConfig(model_path="meta-llama/Llama-3.1-8B-Instruct")
}


def load_model(model_cfg: ModelConfig, model_type, device="cuda:0"):
    device = torch.device(device)
    dtype = getattr(torch, model_cfg.dtype)
    torch.set_default_dtype(dtype)

    with device:
        if model_type=="td":
            model = td_model.from_pretrained(
                model_cfg.model_path,
                device_map=device,
                torch_dtype=dtype,
            )
        elif model_type=="hf":
            model = hf_model.from_pretrained(
                model_cfg.model_path,
                device_map=device,
                torch_dtype=dtype,
            )
    return model


@torch.inference_mode()
def benchmark_tidal():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CFGS.keys(), default="llama3.1-8b-instruct")
    parser.add_argument("--context_len", type=int, default=2 * 1024)
    parser.add_argument("--decode_len", type=int, default=256)
    parser.add_argument("--page_size", type=int, default=1)
    parser.add_argument("--token_budget", type=int, default=256)
    parser.add_argument("--iteration", type=int, default=10)
    args = parser.parse_args()

    assert args.model in MODEL_CFGS, f"Model {args.model} not found in MODEL_CFGS"
    model_cfg = MODEL_CFGS[args.model]

    max_seq_len = args.context_len + args.decode_len + 512
    page_size = args.page_size
    token_budget = args.token_budget
    context_len = args.context_len
    decode_len = args.decode_len

    td_model = load_model(model_cfg, model_type="td", device="cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_path)

    dtype = getattr(torch, model_cfg.dtype)
    device = td_model.device
    td_model.tidal_init(
        page_size=page_size,
        max_seq_len=max_seq_len,
        token_budget=token_budget,
        dtype=dtype,
        device=device,
    )

    for _ in tqdm(range(args.iteration)):
        # clear cuda cache
        torch.cuda.empty_cache()

        # Prefill Stage
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {
                "role": "user", 
                "content": "How many helicopters can a human eat in one sitting?"
            },
        ]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")

        response = tokenized_chat.detach().clone()
        output = td_model(input_ids=tokenized_chat)

        for i in range(decode_len):
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
            response = torch.cat([response, next_token], dim=-1)
            output = td_model(
                input_ids=next_token,
            )

        td_model.tidal_clear()

        print(tokenizer.decode(response[0]))


    avg_prefill_latency = np.mean(td_model.model.iController.prefill_latency)
    avg_decode_latency = np.mean(td_model.model.iController.decode_latency)

    print(
        "page_size,token_budget,context_len,decode_len,avg_prefill_latency,avg_decode_latency"
    )
    print(
        f"{page_size},{token_budget},{context_len},{decode_len},{avg_prefill_latency},{avg_decode_latency}"
    )


if __name__ == "__main__":
    benchmark_tidal()

# nsys profile --delay 20 --duration 1 --output "$(env TZ='US/Pacific' date +%Y%m%d-%H%M%S).nsys-rep" python text_gen.py
