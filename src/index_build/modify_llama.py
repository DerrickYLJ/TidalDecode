import math
import time  # Import time module
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
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
import faiss
import numpy as np

def latency_print(update_time, search_time, sdpa_time, q_len):
    if q_len != 1:
        print("prefilling: ")
        print(f"q_len: {q_len}")
        print(f"Index Creation Time: {update_time:.6f} seconds" if update_time is not None else "Index Creation Time: 0.0 seconds")
        print(f"Index Update Time: {search_time:.6f} seconds" if search_time is not None else "Index Update Time: 0.0 seconds")
    else:
        print(f"q_len: {q_len}")
        print(f"Index Update Time: {update_time:.6f} seconds" if update_time is not None else "Index Update Time: 0.0 seconds")
        print(f"Index Search Time: {search_time:.6f} seconds" if search_time is not None else "Index Search Time: 0.0 seconds")
    print(f"sdpa Time: {sdpa_time:.6f} seconds")

class KeyValueIndexStore:
    def __init__(self, res, dimension, num_kv_heads, top_k, layer_idx):
        self.dimension = dimension
        self.num_kv_heads = num_kv_heads
        self.key_index_store = [
            faiss.IndexFlatL2(dimension) for _ in range(num_kv_heads)  # on CPU
        ]
        self.key_index_store = [
            faiss.index_cpu_to_gpu(res, 0, index) for index in self.key_index_store
        ]
        self.top_k = top_k
        self.layer_idx = layer_idx

    def update_index_store(self, key_states):
        bsz, num_kv_heads, q_len, head_dim = key_states.size()
        if num_kv_heads != self.num_kv_heads:
            raise ValueError(
                f"dimension of key_states when updating index store is wrong: should be {self.num_kv_heads} but got {num_kv_heads}"
            )
        for head in range(self.num_kv_heads):
            keys = key_states[:, head, :, :].reshape(-1, head_dim).numpy()
            keys = np.ascontiguousarray(keys).astype(np.float32)
            self.key_index_store[head].add(keys)

    def batch_search_index_store(self, query_states, past_key_value):
        # batch the query heads in the search for each kv_head
        if self.top_k is None:
            raise ValueError("Top-k is None!")
        bsz, num_heads, q_len, head_dim = query_states.size()
        self.kv_group_num = num_heads // self.num_kv_heads
        retrieved_k = []
        retrieved_v = []
        bsz, num_heads, q_len, head_dim = query_states.size()
        if q_len > 1:
            raise ValueError("Index Retrieval only supports generation!")
        
        # Perform batched retrieval for each key-value head
        for i_index in range(self.num_kv_heads):
            # Gather queries for the current key-value head
            head_indices = np.arange(i_index*self.kv_group_num, (i_index+1)*self.kv_group_num)
            queries = query_states[:, head_indices, :, :].reshape(-1, head_dim).numpy()
            queries = np.ascontiguousarray(queries).astype(np.float32)
            
            # Perform the batched search on the current index store
            _, I_k = self.key_index_store[i_index].search(queries, k=self.top_k)
            # Sort the indices and retrieve keys and values
            sorted_indices = np.argsort(I_k, axis=1)
            sorted_I_k = np.take_along_axis(I_k, sorted_indices, axis=1)
            keys_retrieved = past_key_value.key_cache[self.layer_idx][:, i_index, :, :].reshape(-1, head_dim)[sorted_I_k].reshape(bsz, -1, self.top_k, head_dim)
            values_retrieved = past_key_value.value_cache[self.layer_idx][:, i_index, :, :].reshape(-1, head_dim)[sorted_I_k].reshape(bsz, -1, self.top_k, head_dim)
            
            retrieved_k.append(keys_retrieved)
            retrieved_v.append(values_retrieved)
        
        retrieved_k = torch.cat(retrieved_k, dim=1)
        retrieved_v = torch.cat(retrieved_v, dim=1)

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
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    top_k: int = None,
    res = None,
    **kwargs,
):
    
    if output_attentions:
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    search_time = None
    update_time = None
    spda_time = None

    if top_k != 0:
        if q_len == 1:
            start_time = time.time()
            key_states_cpu = key_states.cpu()
            self.kv_index_store.update_index_store(key_states_cpu[:, :, [-1], :])
            update_time = time.time() - start_time
            start_time = time.time()
            query_states_cpu = query_states.cpu()
            key_states, value_states = self.kv_index_store.batch_search_index_store(query_states_cpu, past_key_value)
            search_time = time.time() - start_time
        else:
            start_time = time.time()
            self.kv_index_store = KeyValueIndexStore(
                res, self.head_dim, self.num_key_value_heads, top_k, self.layer_idx
            )
            update_time = time.time() - start_time
            start_time = time.time()
            key_states_cpu = key_states.cpu()
            self.kv_index_store.update_index_store(key_states_cpu) # parallelizable 
            search_time = time.time() - start_time
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
    else:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False
    torch.cuda.synchronize()
    start_time = time.time()
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    torch.cuda.synchronize()
    spda_time = time.time() - start_time
    

    attn_output = attn_output.transpose(1, 2).contiguous()
   
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)
    latency_print(update_time, search_time, spda_time, q_len)
    return attn_output, None, past_key_value

def enable_llama_index_build_attention(model, top_k, comm=None, res=None):
    res = faiss.StandardGpuResources() if res is None else res
    def wrap_forward(module):

        def new_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None, 
            top_k=top_k,
            res=res,
            **kwargs,
        ):
            return llama_index_build_attention_forward(
                module,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                top_k=top_k,
                res=res,
                **kwargs,
            )

        module.forward = new_forward

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_index_build_attention(module, top_k, comm, res)
        if isinstance(module, LlamaAttention):
            wrap_forward(module)
