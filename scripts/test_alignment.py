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
from tidal.utils import apply_llama31_rope_in_place
from src.utils import load, download_url, load_jsonl

def check_tensor_alignment(a, b):
    assert a.shape == b.shape, f"Tensor shape is not aligned with a:{a.shape}, b:{b.shape}"
    torch.testing.assert_close(actual=a, expected=b, check_device=False)

@dataclasses.dataclass
class ModelConfig:
    """
    ModelConfig is a configuration class for setting up the model parameters.

    Attributes:
        model_path (str): The path to the model file.
        dtype (str): The data type to be used for the model. Default is "float16".
        device (str): The device to run the model on. Default is "cuda:0".
    """
    model_path: str
    dtype: str = dataclasses.field(default="float16")
    device: str = dataclasses.field(default="cuda:0")


MODEL_CFGS = {
    "llama2-7b": ModelConfig(model_path="meta-llama/Llama-2-7b-chat-hf"),
    "llama3.1-8b": ModelConfig(model_path="meta-llama/Llama-3.1-8B"),
    "llama3.1-8b-instruct": ModelConfig(model_path="meta-llama/Llama-3.1-8B-Instruct")
}


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    assert args.model in MODEL_CFGS, f"Model {args.model} not found in MODEL_CFGS"
    model_cfg = MODEL_CFGS[args.model]

    max_seq_len = args.context_len + args.decode_len + 512
    page_size = args.page_size
    token_budget = args.token_budget
    context_len = args.context_len
    decode_len = args.decode_len

    td_model = load_model(model_cfg, model_type="td", device="cuda:0")
    ref_model, tokenizer = load(model_cfg.model_path, attn_type="tidal", device="cuda:1", top_k=token_budget, sparse_layer_start=2, correction_layer=13)
    hidden_size = td_model._config.hidden_size

    # tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_path)

    dtype = getattr(torch, model_cfg.dtype)
    device = td_model.device
    td_model.tidal_init(
        page_size=page_size,
        max_seq_len=max_seq_len,
        token_budget=token_budget,
        dtype=dtype,
        device=device,
    )

    # clear cuda cache
    torch.cuda.empty_cache()

    # Prefill Stage
    question = "How many helicopters can a human eat in one sitting?"
    # question = "How are you doing?"
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        # {
        #     "role": "user", 
        #     "content": "How many helicopters can a human eat in one sitting?"
        # },
        {
            "role": "user", 
            "content": question
        },
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    prompt_len = tokenized_chat.shape[-1]
    ref_output = ref_model.generate(input_ids=tokenized_chat.to(ref_model.device), max_length=256, do_sample=False)
    td_output = td_model.generate(input_ids=tokenized_chat.to(td_model.device), max_length=256, do_sample=False)
    print("question:", question)
    print("ref_output:", tokenizer.decode(ref_output[0, prompt_len:]))
    print("-"*100)
    print("td_output:", tokenizer.decode(td_output[0, prompt_len:]))
    exit(0)
    ref_output_text = tokenized_chat.to(ref_model.device)
    td_output_text = tokenized_chat.to(td_model.device)

    with torch.no_grad():
        ref_output = ref_model(input_ids=tokenized_chat.to(ref_model.device), use_cache=True)
        td_output = td_model(input_ids=tokenized_chat.to(td_model.device))

        for i in range(100):
            ref_next_token = ref_output.logits[:, -1].argmax(dim=-1, keepdim=True)
            td_next_token = td_output.logits[:, -1].argmax(dim=-1, keepdim=True)
            ref_output_text = torch.cat([ref_output_text, ref_next_token], dim=-1)
            td_output_text = torch.cat([td_output_text, td_next_token], dim=-1)
            # print(ref_next_token, td_next_token)
            assert ref_next_token.detach().item() == td_next_token.detach().item(), f"Next token mismatch at step {i}"
            # print(tokenizer.decode(ref_output_text[0]))
            # print(tokenizer.decode(td_output_text[0]))
            td_output = td_model(input_ids=td_next_token)
            ref_output = ref_model(input_ids=ref_next_token, use_cache=True, past_key_values=ref_output.past_key_values)
            # if i == 71:
            #     exit(0)
    print(tokenizer.decode(ref_output_text[0]))
    print(tokenizer.decode(td_output_text[0]))
    


if __name__ == "__main__":
    benchmark_tidal()

# nsys profile --delay 20 --duration 1 --output "$(env TZ='US/Pacific' date +%Y%m%d-%H%M%S).nsys-rep" python text_gen.py
