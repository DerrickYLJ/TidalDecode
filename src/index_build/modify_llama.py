import math
from typing import Optional, Tuple
import torch.nn.functional as F
from torch import nn

import torch

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)

import faiss
import numpy as np


class KeyValueIndexStore:
    def __init__(self, dimension, num_kv_heads, top_k, prefilling):
        self.dimension = dimension
        self.num_kv_heads = num_kv_heads
        self.key_index_store = [
            faiss.IndexFlatL2(dimension) for _ in range(num_kv_heads)  # on CPU
        ]
        self.value_index_store = [
            faiss.IndexFlatL2(dimension) for _ in range(num_kv_heads)  # on CPU
        ]
        self.top_k = top_k
        self.past_key_value = prefilling
    
    def update_kv_cache(self, key_states, value_states):
        self.past_key_value[0] = torch.cat([self.past_key_value[0], key_states], dim=2)
        self.past_key_value[1] = torch.cat([self.past_key_value[1], value_states], dim=2)
        return (self.past_key_value[0], self.past_key_value[1])

    def update_index_store(self, key_states, value_states):
        # key_states and value_states are of shape (bsz, num_key_value_heads, q_len, head_dim)
        bsz, num_kv_heads, q_len, head_dim = key_states.size()
        if num_kv_heads != self.num_kv_heads:
            raise ValueError(
                f"dimension of key_states when updating index store is wrong: should be {self.num_kv_heads} but got {num_kv_heads}"
            )
        if key_states.size() != value_states.size():
            raise ValueError(
                f"mismatch dimension of key_states and value_states when updating index store"
            )
        for head in range(self.num_kv_heads):
            # Convert to numpy for FAISS
            keys = key_states[:, head, :, :].reshape(-1, head_dim).numpy()
            values = value_states[:, head, :, :].reshape(-1, head_dim).numpy()
            keys = np.ascontiguousarray(keys).astype(np.float32)
            values = np.ascontiguousarray(values).astype(np.float32)
            self.key_index_store[head].add(keys)
            self.value_index_store[head].add(values)

    def search_index_store(self, query_states):
        if self.top_k is None:
            raise ValueError("Top-k is None!")
        bsz, num_heads, q_len, head_dim = query_states.size()
        retrieved_k = []
        retrieved_v = []
        if q_len > 1:
            raise ValueError("Index Retrieval only supports generation!")
        for head in range(num_heads):
            i_index = head // (num_heads // self.num_kv_heads)
            queries = query_states[:, head, :, :].reshape(-1, head_dim).numpy()
            queries = np.ascontiguousarray(queries).astype(np.float32)
            D_k, I_k = self.key_index_store[i_index].search(queries, k=self.top_k)
            sorted_indices = np.argsort(I_k, axis=1)
            sorted_I_k = np.take_along_axis(I_k, sorted_indices, axis=1)
            # D_v, I_v = self.value_index_store[i_index].search(queries, k=self.top_k)
            keys_retrieved = self.past_key_value[0][:, i_index, :, :].reshape(-1, head_dim)[sorted_I_k].reshape(bsz, self.top_k, head_dim)
            values_retrieved = self.past_key_value[1][:, i_index, :, :].reshape(-1, head_dim)[sorted_I_k].reshape(bsz, self.top_k, head_dim)
            retrieved_k.append(keys_retrieved)
            retrieved_v.append(values_retrieved)
        retrieved_k = torch.stack(
            retrieved_k
        )  # Shape: (bsz, num_heads, self.top_k, head_dim)
        retrieved_v = torch.stack(
            retrieved_v
        )  # Shape: (bsz, num_heads, self.top_k, head_dim)
        retrieved_k = retrieved_k.view(num_heads, bsz, self.top_k, head_dim).transpose(0, 1)
        retrieved_v = retrieved_v.view(num_heads, bsz, self.top_k, head_dim).transpose(0, 1)
        if retrieved_k.size() != (bsz, num_heads, self.top_k, head_dim):
            raise ValueError(
                f"retrieved shape is incorrect, should be ({bsz, num_heads, self.top_k, head_dim}), but got {retrieved_k.size()}"
            )
        if retrieved_k.size() != retrieved_v.size():
            raise ValueError(
                f"retrieved_k and retrieved_v are mismatched, retrieved_k: {retrieved_k.size()} but retrieved_v: {retrieved_v.size()}"
            )
        
        return (retrieved_k, retrieved_v)


def llama_index_build_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    top_k: int = None,
):
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += self.kv_index_store.past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    
    # update index_store + kv_cache
    key_states_cpu = key_states.to("cpu")
    value_states_cpu = value_states.to("cpu")

    if past_key_value is not None:
        key_states, value_states = self.kv_index_store.update_kv_cache(key_states_cpu, value_states_cpu)
        if key_states.size() != value_states.size():
            raise ValueError(
                f"key_states and value_states are mismatched, key_states: {key_states.size()} but value_states: {value_states.size()}"
            )
        past_key_value = (self.kv_index_store.past_key_value[0], self.kv_index_store.past_key_value[1])
    else:
        self.kv_index_store = KeyValueIndexStore(
            self.head_dim, self.num_key_value_heads, top_k, [key_states_cpu, value_states_cpu]
        )
        past_key_value = (key_states_cpu, value_states_cpu)

    self.kv_index_store.update_index_store(key_states_cpu, value_states_cpu)

    # select top_k keys and values from kv_cache based on index_store
    if top_k is not None and q_len == 1:
        # retrieve top-k in the index store
        key_states, value_states = self.kv_index_store.search_index_store(
            query_states.to("cpu")
        )
        key_states = key_states.to(hidden_states.device)
        value_states = value_states.to(hidden_states.device)
    else:
        key_states, value_states = key_states.to(hidden_states.device), value_states.to(hidden_states.device)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if top_k is not None:
        if q_len > 1:
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
        else:
            if attn_weights.size() != (bsz, self.num_heads, q_len, top_k):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, top_k)}, but is"
                    f" {attn_weights.size()}"
                )
    else:
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    # print(f"{self.layer_idx}: {kv_seq_len}: hidden_states: {hidden_states.device}; query_states: {query_states.device}; key_states: {key_states.device}; attn_output: {attn_output.device}")
    return attn_output, attn_weights, past_key_value

def llama_index_build_dist_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    top_k: int = None,
):
    bsz, q_len, _ = hidden_states.size()
    device_count = torch.cuda.device_count()
    default_device = hidden_states.device

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None
    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #     raise ValueError(
    #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
    #         f" {attn_weights.size()}"
    #     )
    # attn_weights_ground_truth_1 = attn_weights
    # if attention_mask is not None:
    #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    #         raise ValueError(
    #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
    #         )
    #     attn_weights_ground_truth_2 = attn_weights + attention_mask

    # attn_weights_ground_truth_3 = nn.functional.softmax(attn_weights_ground_truth_2, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # attn_weights_ground_truth_4 = torch.matmul(attn_weights_ground_truth_3, value_states)

    # if attn_weights_ground_truth_4.size() != (bsz, self.num_heads, q_len, self.head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )
    # attn_weights_ground_truth_4 = attn_weights_ground_truth_4.transpose(1, 2).contiguous()
    print(query_states.shape[-2])
    if query_states.shape[-2] > 1:
        # split keys
        key_states_split = [key_states.chunk(device_count, dim=-2)[i].to(f'cuda:{i}') for i in range(device_count)]
        attention_mask_split = [attention_mask.chunk(device_count, dim=-2)[i].to(f'cuda:{i}') for i in range(device_count)]if attention_mask is not None else [None] * device_count
        del key_states
        del attention_mask

        # tensor parallelism
        # split attn_weights
        attn_weights = []
        for i in range(device_count):
            with torch.cuda.device(i):
                key_chunk = key_states_split[i].to(f'cuda:{i}')
                query_copy = query_states.to(f'cuda:{i}')
                attn_weights_chunk = torch.matmul(query_copy, key_chunk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn_weights.append(attn_weights_chunk)
                del key_chunk, query_copy  # Free up memory
        attn_weights_subchunks = [torch.chunk(attn, device_count, dim=-2) for attn in attn_weights] # 4 chunks, each has 4 subchunks
        
        # reorder attn_weights
        reordered_attn_weights = []
        for i in range(device_count):
            reordered_subchunks = [attn_weights_subchunks[j][i].to(f'cuda:{i}') for j in range(device_count)]
            reordered_attn_weights.append(torch.cat(reordered_subchunks, dim=-1))
            for subchunk in reordered_subchunks:
                del subchunk  # Free up memory
        # ### test 1 ###
        # attn_weights_test_1 = torch.cat([reordered.to(default_device) for reordered in reordered_attn_weights], dim=-2)
        # assert(torch.equal(attn_weights_test_1, attn_weights_ground_truth_1), "test 1")
        # ### test 1 ###

        # attention mask
        attn_weights_masked = []
        for i in range(device_count):
            with torch.cuda.device(i):
                attn_weights_with_mask = reordered_attn_weights[i]
                attn_weights_with_mask += attention_mask_split[i]
                attn_weights_masked.append(attn_weights_with_mask)
        for i in range(device_count):
            del reordered_attn_weights[0]
        # ### test 2 ###
        # attn_weights_test_2 = torch.cat([masked.to(default_device) for masked in attn_weights_masked], dim=-2)
        # assert torch.equal(attn_weights_test_2, attn_weights_ground_truth_2), "test 2"
        # ### test 2 ###

        # split softmax
        attn_weights_softmax = []
        for i in range(device_count):
            with torch.cuda.device(i):
                softmax_chunk = nn.functional.softmax(attn_weights_masked[i], dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_softmax.append(softmax_chunk)
        for i in range(device_count):
            del attn_weights_masked[0]
        # ### test 3 ###
        # softmax_test_3 = torch.cat([soft.to(default_device) for soft in attn_weights_softmax], dim=-2)
        # assert torch.equal(softmax_test_3, attn_weights_ground_truth_3), "test 3"
        # ### test 3 ###
        
        # attention output
        attn_output = []
        for i in range(device_count):
            with torch.cuda.device(i):
                value_chunk = value_states.to(f'cuda:{i}')
                output_chunk = torch.matmul(attn_weights_softmax[i], value_chunk)
                attn_output.append(output_chunk.to(default_device))
                del output_chunk, value_chunk  # Free up memory
        for i in range(device_count):
            del attn_weights_softmax[0]
        attn_output = torch.cat(attn_output, dim=2).transpose(1, 2).contiguous()

        # ### test 4 ###
        # assert torch.equal(attn_output, attn_weights_ground_truth_4), "test 4"
        # ### test 4 ###
    else:
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_llama_index_build_attention(model, top_k):
    def wrap_forward(module):
        original_forward = module.forward

        def new_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            top_k=top_k,
        ):
            return llama_index_build_dist_attention_forward(
                module,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                top_k=top_k,
            )

        module.forward = new_forward

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_index_build_attention(module, top_k)
        if isinstance(module, LlamaAttention):
            wrap_forward(module)
