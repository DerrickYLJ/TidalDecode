# Based on Punica Project
# Check: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/benchmarks/bench_textgen.py

import argparse
import dataclasses
import time
import numpy as np
import torch
from tqdm.auto import tqdm

from tidal import LlamaForCausalLM


@dataclasses.dataclass
class ModelConfig:
    model_path: str
    dtype: str = dataclasses.field(default="float16")
    device: str = dataclasses.field(default="cuda:0")


MODEL_CFGS = {
    "llama2-7b": ModelConfig(model_path="meta-llama/Llama-2-7b-chat-hf"),
}


def load_model(model_cfg: ModelConfig):
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    torch.set_default_dtype(dtype)

    with device:
        model = LlamaForCausalLM.from_pretrained(
            model_cfg.model_path,
            device_map=device,
            torch_dtype=dtype,
        )
    return model


@torch.inference_mode()
def benchmark_tidal():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CFGS.keys(), default="llama2-7b")
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

    model = load_model(model_cfg)

    dtype = getattr(torch, model_cfg.dtype)
    device = torch.device(model_cfg.device)
    model.tidal_init(
        page_size=page_size,
        max_seq_len=max_seq_len,
        token_budget=token_budget,
        dtype=dtype,
        device=device,
    )
    hidden_size = model._config.hidden_size

    for _ in tqdm(range(args.iteration)):
        # clear cuda cache
        torch.cuda.empty_cache()

        # Prefill Stage
        hidden_states = torch.randn(
            1, context_len, hidden_size, dtype=dtype, device=device
        )
        model(
            inputs_embeds=hidden_states,
        )
        # Start decoding decode_len tokens
        for _ in range(decode_len):
            hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
            model(
                inputs_embeds=hidden_states,
            )

        model.tidal_clear()

    avg_prefill_latency = np.mean(model.model.iController.prefill_latency)
    avg_decode_latency = np.mean(model.model.iController.decode_latency)

    print(
        "page_size,token_budget,context_len,decode_len,avg_prefill_latency,avg_decode_latency"
    )
    print(
        f"{page_size},{token_budget},{context_len},{decode_len},{avg_prefill_latency},{avg_decode_latency}"
    )


if __name__ == "__main__":
    benchmark_tidal()

# nsys profile --delay 20 --duration 1 --output "$(env TZ='US/Pacific' date +%Y%m%d-%H%M%S).nsys-rep" python text_gen.py
