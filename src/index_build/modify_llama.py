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
    def __init__(self, res, dimension, num_kv_heads, top_k, layer_idx, sparse_layer_start):
        self.dimension = dimension
        self.num_kv_heads = num_kv_heads
        self.sparse_layer_start = sparse_layer_start
        # self.key_index_store = [
        #     faiss.index_cpu_to_gpu(res, 0, index) for index in self.key_index_store
        # ]
        self.retrieve_dict = {i: {} for i in range(self.num_kv_heads)}
        self.top_k = top_k
        self.layer_idx = layer_idx
        if self.layer_idx == sparse_layer_start or self.layer_idx == 15:
            self.key_index_store = [
                faiss.IndexFlatIP(dimension) for _ in range(num_kv_heads)  # on CPU
            ]

    def update_index_store(self, key_states):
        if self.layer_idx == self.sparse_layer_start or self.layer_idx == 15:
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

    def batch_search_index_store(self, query_states, past_key_value, pos_dict):
        if self.top_k is None:
            raise ValueError("Top-k is None!")
        
        bsz, num_heads, q_len, head_dim = query_states.size()
        self.kv_group_num = num_heads // self.num_kv_heads

        if q_len > 1:
            raise ValueError("Index Retrieval only supports generation!")

        retrieved_k = []
        retrieved_v = []
        
        # Perform batched retrieval for each key-value head
        for i_index in range(self.num_kv_heads):
            head_indices = np.arange(
                i_index * self.kv_group_num, (i_index + 1) * self.kv_group_num
            )
            if len(pos_dict[i_index]) == 0:
                # Gather queries for the current key-value head
                queries = query_states[:, head_indices, :, :].reshape(-1, head_dim).numpy()
                queries = np.ascontiguousarray(queries).astype(np.float32)

                # Perform the batched search on the current index store
                num_stored_vectors = self.key_index_store[i_index].ntotal
                if num_stored_vectors < self.top_k:
                    # Retrieve all stored vectors by setting k to num_stored_vectors
                    _, I_k = self.key_index_store[i_index].search(queries, k=num_stored_vectors)
                else:
                    # Perform normal top-k retrieval
                    _, I_k = self.key_index_store[i_index].search(queries, k=self.top_k)
                
                sorted_indices = np.argsort(I_k, axis=1)
                sorted_I_k = np.take_along_axis(I_k, sorted_indices, axis=1)

                # Convert indices to torch tensors
                sorted_I_k = torch.from_numpy(sorted_I_k).long()

                # Update the retrieve dictionary
                for i in sorted_I_k[0, :].tolist():  
                    self.retrieve_dict[i_index][i] = self.retrieve_dict[i_index].get(i, 0) + 1 

                pos_dict[i_index] = sorted_I_k

                num_vectors_to_retrieve = sorted_I_k.size(1)
                keys_retrieved = (
                    past_key_value.key_cache[self.layer_idx][:, i_index, :, :]
                    .reshape(-1, head_dim)[sorted_I_k]
                    .reshape(bsz, -1, num_vectors_to_retrieve, head_dim)
                )
                values_retrieved = (
                    past_key_value.value_cache[self.layer_idx][:, i_index, :, :]
                    .reshape(-1, head_dim)[sorted_I_k]
                    .reshape(bsz, -1, num_vectors_to_retrieve, head_dim)
                )
            else:
                sorted_I_k = pos_dict[i_index]
                num_vectors_to_retrieve = sorted_I_k.size(1)

                keys_retrieved = (
                    past_key_value.key_cache[self.layer_idx][:, i_index, :, :]
                    .reshape(-1, head_dim)[sorted_I_k]
                    .reshape(bsz, -1, num_vectors_to_retrieve, head_dim)
                )
                values_retrieved = (
                    past_key_value.value_cache[self.layer_idx][:, i_index, :, :]
                    .reshape(-1, head_dim)[sorted_I_k]
                    .reshape(bsz, -1, num_vectors_to_retrieve, head_dim)
                )

            retrieved_k.append(keys_retrieved)
            retrieved_v.append(values_retrieved)

        retrieved_k = torch.cat(retrieved_k, dim=1)
        retrieved_v = torch.cat(retrieved_v, dim=1)

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
    ] = None,  
    top_k: int = None,
    res=None,
    **kwargs,
):
    sparse_layer_start = 2
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
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    kv_seq_len = past_key_value.get_seq_length(self.layer_idx)
    if self.layer_idx >= sparse_layer_start:
        key_states_cpu = key_states.cpu()
        if q_len != kv_seq_len:
            # pass prefilling phase
            query_states_cpu = query_states.cpu()
            self.kv_index_store.update_index_store(key_states_cpu[:, :, [-1], :])
            key_states, value_states = self.kv_index_store.batch_search_index_store(
                query_states_cpu, past_key_value, self.pos_dict
            )
        else:
            self.kv_index_store = KeyValueIndexStore(
                res, self.head_dim, self.num_key_value_heads, top_k, self.layer_idx, sparse_layer_start
            )
            self.kv_index_store.update_index_store(key_states_cpu)

    if self.layer_idx < sparse_layer_start or q_len == kv_seq_len:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

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
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:
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

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by chunk_size
    seq_length = attn_weights.shape[-1]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // chunk_size,
        chunk_size,
    ).amax(dim=-1)

    _, topk = chunk_attn_weights.topk(
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    )
    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device)
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom

def llama_quest_attention_forward(
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
    self.chunk_size = 16
    self.token_budget = top_k
    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_idx < 2: # TODO: should be self.layer_idx < 2
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.size(2)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states * sign
    postive_query = query_states * sign

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // self.chunk_size,
        self.chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        postive_query.float(),
        chunk_max_key.transpose(2, 3),
    )

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
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
        quantized_weight = quantized_weight + attention_mask
        quantized_weight = torch.max(
            quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
        )

    token_budget = min(kv_seq_len, self.token_budget)

    attn_weights_for_selection = quantized_weight

    if token_budget > 0:
        mask_bottom = local_heavy_hitter_mask(
            attn_weights_for_selection, token_budget, self.chunk_size
        )  # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)
    mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value



def enable_llama_index_build_attention(model, top_k, comm=None, attn_type="index", res=None):
    # pos_dict: kv_head -> indices
    # res = faiss.StandardGpuResources() if res is None else res
    res = None
    print("top_k", top_k, attn_type)

    def wrap_forward(module):
        
        def new_index_forward(
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
        
        def new_quest_forward(
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
            return llama_quest_attention_forward(
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
        if attn_type == "index":
            print("ours")
            module.forward = new_index_forward
        else:
            print("quest")
            module.forward = new_quest_forward

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_index_build_attention(module, top_k, comm, attn_type, res)
        if isinstance(module, LlamaAttention):
            wrap_forward(module)