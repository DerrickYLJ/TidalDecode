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


class KeyValueIndexStore:
    def __init__(self, res, dimension, num_kv_heads, top_k, layer_idx):
        self.dimension = dimension
        self.num_kv_heads = num_kv_heads
        self.key_index_store = [
            faiss.IndexFlatL2(dimension) for _ in range(num_kv_heads)  # on CPU
        ]
        # self.key_index_store = [
        #     faiss.index_cpu_to_gpu(res, 0, index) for index in self.key_index_store
        # ]
        self.top_k = top_k
        self.layer_idx = layer_idx

    def update_index_store(self, key_states):
        bsz, num_kv_heads, q_len, head_dim = key_states.size()
        if num_kv_heads != self.num_kv_heads:
            raise ValueError(
                f"dimension of key_states when updating index store is wrong: should be {self.num_kv_heads} but got {num_kv_heads}"
            )
        for head in range(self.num_kv_heads):
            index = self.key_index_store[head]
            keys = key_states[:, head, :, :].reshape(-1, head_dim).numpy()
            keys = np.ascontiguousarray(keys).astype(np.float32)
            index.add(keys)

    def batch_search_index_store(self, query_states, past_key_value):
        # batch the query heads in the search for each kv_head
        if self.top_k is None:
            raise ValueError("Top-k is None!")
        kv_seqlen = past_key_value.key_cache[self.layer_idx].size(-2)
        bsz, num_heads, q_len, head_dim = query_states.size()
        self.kv_group_num = num_heads // self.num_kv_heads

        bsz, num_heads, q_len, head_dim = query_states.size()
        if q_len > 1:
            raise ValueError("Index Retrieval only supports generation!")

        retrieved_k = []
        retrieved_v = []
        # Perform batched retrieval for each key-value head
        for i_index in range(self.num_kv_heads):
            head_indices = np.arange(
                i_index * self.kv_group_num, (i_index + 1) * self.kv_group_num
            )

            # Gather queries for the current key-value head
            queries = query_states[:, head_indices, :, :].reshape(-1, head_dim).numpy()
            queries = np.ascontiguousarray(queries).astype(np.float32)

            # Perform the batched search on the current index store
            _, I_k = self.key_index_store[i_index].search(queries, k=self.top_k)
            sorted_indices = np.argsort(I_k, axis=1)
            sorted_I_k = np.take_along_axis(I_k, sorted_indices, axis=1)

            # Efficiently gather and assign the top-k key and value states
            sorted_I_k = torch.from_numpy(
                sorted_I_k
            ).long()  # Convert indices to torch tensors

            keys_retrieved = (
                past_key_value.key_cache[self.layer_idx][:, i_index, :, :]
                .reshape(-1, head_dim)[sorted_I_k]
                .reshape(bsz, -1, self.top_k, head_dim)
            )
            values_retrieved = (
                past_key_value.value_cache[self.layer_idx][:, i_index, :, :]
                .reshape(-1, head_dim)[sorted_I_k]
                .reshape(bsz, -1, self.top_k, head_dim)
            )

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
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.45
    top_k: int = None,
    res=None,
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

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    if self.layer_idx > 2:
        # layer_idx <= 2 will be critical to generation quality
        key_states_cpu = key_states.cpu()
        if q_len == 1:
            # inference
            query_states_cpu = query_states.cpu()
            self.kv_index_store.update_index_store(key_states_cpu[:, :, [-1], :])
            key_states, value_states = self.kv_index_store.batch_search_index_store(
                query_states_cpu, past_key_value
            )
        else:
            self.kv_index_store = KeyValueIndexStore(
                res, self.head_dim, self.num_key_value_heads, top_k, self.layer_idx
            )
            self.kv_index_store.update_index_store(key_states_cpu)  # parallelizable

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if self.layer_idx <= 2 or q_len != 1:
        # prefilling or small layer
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value
    else:
        # generation
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


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

        module.flash_forward = module.forward
        module.forward = new_forward

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_index_build_attention(module, top_k, comm, res)
        if isinstance(module, LlamaAttention):
            wrap_forward(module)
