import torch
import argparse
import os.path as osp
import ssl
import urllib.request
import os
import json
from src.enable_tidal import enable_tidal


def parse_args():
    # TODO: modify arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/llama/llama-7b"
    )
    parser.add_argument(
            "--attn_type",
                type=str,
                choices=[
                    "tidal",
                ],
                default="index",
        )
    parser.add_argument("--", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")

    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    args = parser.parse_args()
    return args

def load(model_name_or_path, attn_type, device, **kwargs):
    print(f"Loading model from {model_name_or_path} ...")
    
    if attn_type=="tidal":
        
        print("TidalDecode enabled!")
        from transformers import (
            AutoTokenizer,
        )
        if "Yarn" in model_name_or_path:
            from src.models.yarn_tidaldecoding import (
                LlamaForCausalLM
            )
        else:
            from src.models.llama_tidaldecoding import (
                LlamaForCausalLM
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        enable_tidal(model, attn_type, **kwargs)
    else:
        # flash attention
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )
        print("full-weight attention enabled")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    # print(f"Loaded Model: {model}")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer



def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict

