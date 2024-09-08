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

def latency_output(update_time, search_time, sdpa_time, q_len):
    # if q_len != 1:
    #     # print("prefilling: ")
    #     # print(f"q_len: {q_len}")
    #     # print(f"Index Creation Time: {update_time:.6f} seconds" if update_time is not None else "Index Creation Time: 0.0 seconds")
    #     # print(f"Index Update Time: {search_time:.6f} seconds" if search_time is not None else "Index Update Time: 0.0 seconds")
    # else:
    #     # print(f"q_len: {q_len}")
    #     # print(f"Index Update Time: {update_time:.6f} seconds" if update_time is not None else "Index Update Time: 0.0 seconds")
    #     # print(f"Index Search Time: {search_time:.6f} seconds" if search_time is not None else "Index Search Time: 0.0 seconds")
    # # print(f"sdpa Time: {sdpa_time:.6f} seconds")
    return

class KeyValueIndexStore:
    def __init__(self, res, dimension, num_kv_heads, top_k, layer_idx):
        self.dimension = dimension
        self.num_kv_heads = num_kv_heads
        self.key_index_store = [
            faiss.IndexFlatIP(dimension) for _ in range(num_kv_heads)  # on CPU
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
            before_add = time.time()
            keys = key_states[:, head, :, :].reshape(-1, head_dim).numpy()
            after_trans = time.time()
            # print(f"transform: {after_trans - before_add}")
            keys = np.ascontiguousarray(keys).astype(np.float32)
            # print(f"continuous: {time.time() - after_trans}")
            after_continue = time.time()
            self.key_index_store[head].add(keys)
            # print(f"add: {time.time() - after_continue}")
            

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
        
        retrieved_k = torch.zeros((bsz, num_heads, kv_seqlen, head_dim), dtype=query_states.dtype, device=past_key_value.key_cache[self.layer_idx].device)
        retrieved_v = torch.zeros((bsz, num_heads, kv_seqlen, head_dim), dtype=query_states.dtype, device=past_key_value.value_cache[self.layer_idx].device)
        
        # Perform batched retrieval for each key-value head
        for i_index in range(self.num_kv_heads):
            key_cache = past_key_value.key_cache[self.layer_idx]
            value_cache = past_key_value.value_cache[self.layer_idx]
            head_indices = np.arange(i_index * self.kv_group_num, (i_index + 1) * self.kv_group_num)

            # Gather queries for the current key-value head
            queries = query_states[:, head_indices, :, :].reshape(-1, head_dim).numpy()
            queries = np.ascontiguousarray(queries).astype(np.float32)

            # Perform the batched search on the current index store
            _, I_k = self.key_index_store[i_index].search(queries, k=self.top_k)
            sorted_indices = np.argsort(I_k, axis=1)
            sorted_I_k = np.take_along_axis(I_k, sorted_indices, axis=1)

            # Efficiently gather and assign the top-k key and value states
            sorted_I_k = torch.from_numpy(sorted_I_k).long()  # Convert indices to torch tensors

            for j, head in enumerate(head_indices):
                # Use advanced indexing to assign top-k keys and values efficiently
                retrieved_k[:, head, sorted_I_k[j], :] = key_cache[:, i_index, sorted_I_k[j], :]
                retrieved_v[:, head, sorted_I_k[j], :] = value_cache[:, i_index, sorted_I_k[j], :]

        if retrieved_k.size() != (bsz, num_heads, kv_seqlen, head_dim):
            raise ValueError(
                f"retrieved shape is incorrect, should be ({bsz, num_heads, kv_seqlen, head_dim}), but got {retrieved_k.size()}"
            )
        if retrieved_k.size() != retrieved_v.size():
            raise ValueError(
                f"retrieved_k and retrieved_v are mismatched, retrieved_k: {retrieved_k.size()} but retrieved_v: {retrieved_v.size()}"
            )
        
        return (retrieved_k != 0, retrieved_v != 0)

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
        # updated_kv = past_key_value.update(key_states.cpu(), value_states.cpu(), self.layer_idx, cache_kwargs)
        # key_states, value_states = updated_kv[0].to(query_states.device), updated_kv[1].to(query_states.device)
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
            # updated_kv = self.kv_index_store.batch_search_index_store(query_states_cpu, past_key_value)
            # key_states, value_states = updated_kv[0].to(query_states.device), updated_kv[1].to(query_states.device)
            if self.layer_idx > 2:
                key_states_mask, _ = self.kv_index_store.batch_search_index_store(query_states_cpu, past_key_value)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
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


    if self.layer_idx <= 2 or q_len != 1:
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
        latency_output(update_time, search_time, spda_time, q_len)
        return attn_output, None, past_key_value

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    sparse_mask = torch.matmul(query_states, key_states_mask.to(query_states.dtype).transpose(2, 3))
    epsilon = torch.finfo(query_states.dtype).eps  # Get the smallest positive number for the dtype
    sparse_mask = sparse_mask.abs() < epsilon  # Use a tolerance to check near-zero values
    attn_weights[sparse_mask] = torch.finfo(attn_weights.dtype).min


    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
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
    self.token_budget = 64
    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_idx < 2:
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
