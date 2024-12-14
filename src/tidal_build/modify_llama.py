import math
import time  # Import time module
import types
from typing import Optional, Tuple
import torch.nn.functional as F
from torch import nn
import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaSdpaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward


def llama_tidal_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    top_k: int = None,
    sparse_layer_start=2,
    correction_layer=9,
    **kwargs,
):
    # prefilling: as full-weight attention
    # generation:
    # - non-sparse layers: full-weight attention #1
    # - sparse_layer_start: full-weight attention + top_k selection #2          topk: (1, 32, 1, topk) => fold (1, 32, topk)
    # - sattn_layer_start -> correction layer - 1: use the same top-k #3        Q: (1, 32, 1, 128) K: (1, 32, topk, 128)
    # - correction layer: full-weight attention + new top_k selection
    # - after correction layer: use the same top-k
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
    # query_states = torch.ones_like(query_states).to(query_states.device)
    # key_states = torch.ones_like(key_states).to(key_states.device)  

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    kv_seq_len = past_key_value.get_seq_length(self.layer_idx)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if self.layer_idx < sparse_layer_start or q_len == kv_seq_len:
        # non-sparse layers or prefilling
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
        # generation
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        last_dim_size = attn_weights.size(-1)
        token_budget = min(last_dim_size, top_k)

        # decoding
        if self.layer_idx == sparse_layer_start or self.layer_idx == correction_layer:
            # extract top_k mask
            _, top_k_indices = torch.topk(attn_weights[:, :, :, :-1], k=token_budget-1, dim=-1)
            # exit(0)
            top_k_indices = torch.cat([top_k_indices, torch.tensor([kv_seq_len-1]*self.num_heads).to(top_k_indices.device).reshape(1,self.num_heads,1,1)], dim=-1)
            top_k_mask = torch.zeros_like(attn_weights).scatter_(-1, top_k_indices, 1.0)
            self.pos_dict = top_k_mask  # store top_k mask
            # if self.layer_idx == 2:
            #     print(self.layer_idx, attn_weights[0, 0, 0, :-1])
            # print(self.layer_idx, top_k_indices[0, 0, 0, :])
        else:
            # apply top_k mask
            if self.pos_dict == None:
                raise ValueError("pos dict should be set up in sparse attn layers")
            min_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(
                self.pos_dict.to(attn_weights.device) == 0, min_value
            )
            # print(self.layer_idx, attn_weights[0, 0, 0, :])
            

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        # if self.layer_idx != sparse_layer_start and self.layer_idx != correction_layer:
        #     print(self.layer_idx, attn_weights[0, 0, 0, :].sum())
        #     exit(0)
        # if self.layer_idx == 2:
        #     print(self.layer_idx, query_states[0, 0, 0, :20], key_states[0, 0, -1, :20])

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

def llama_tidal_forward_yarn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    is_padded_inputs: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, h_size = hidden_states.size()

    # Prefill stage utilizes flash attention
    if q_len > 1 or self.layer_id < self.sparse_layer_start:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            is_padded_inputs,
        )

    has_layer_past = past_key_value is not None

    if has_layer_past:
        past_kv = past_key_value[0]
        past_len = past_key_value[1]
    else:
        past_len = 0

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        q = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        q = torch.cat(q, dim=-1)

        k = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        k = torch.cat(k, dim=-1)

        v = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        v = torch.cat(v, dim=-1)

    else:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

    q = q.view(bsz, q_len, self.num_heads, self.head_dim)
    k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    q, k = self.rotary_emb(q, k, past_len)

    @torch.jit.script
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, slen, _, num_key_value_heads, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, :, None, :].expand(
            batch, slen, 2, num_key_value_heads, n_rep, head_dim
        )
        return hidden_states.reshape(
            batch, slen, 2, num_key_value_heads * n_rep, head_dim
        )

    kv = torch.stack([k, v], 2)
    kv = repeat_kv(kv, self.num_key_value_groups)

    # Cache QKV values
    if has_layer_past:
        new_len = past_len + q.size(1)
        if new_len > past_kv.size(1):
            past_kv = torch.cat(
                [
                    past_kv,
                    torch.empty(
                        bsz,
                        256,
                        2,
                        kv.size(3),
                        kv.size(4),
                        dtype=kv.dtype,
                        device=kv.device,
                    ),
                ],
                1,
            )
        past_kv[:, past_len:new_len] = kv
        kv = past_kv[:, :new_len]
    else:
        past_kv = kv

    k, v = kv.split(1, dim=2)
    k = k.squeeze(2)
    v = v.squeeze(2)

    past_key_value = (past_kv, past_len + q.size(1)) if use_cache else None

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

    last_dim_size = attn_weights.size(-1)
    token_budget = min(last_dim_size, self.token_budget)
    if self.layer_id == self.sparse_layer_start or self.layer_id == self.correction_layer:
        # extract top_k mask
        _, top_k_indices = torch.topk(attn_weights, k=token_budget, dim=-1)
        top_k_mask = torch.zeros_like(attn_weights).scatter_(-1, top_k_indices, 1.0)
        self.pos_dict = top_k_mask # store top_k mask
    else:
        # apply top_k mask
        if self.pos_dict == None:
            raise ValueError("pos dict should be set up in sparse attn layers"
            )
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(self.pos_dict.to(attn_weights.device) == 0, min_value)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q.dtype
    )
    attn_output = torch.matmul(attn_weights, v)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

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

global layer_id
layer_id = 32


def enable_llama_tidal_attention(
    model, top_k, attn_type="tidal", sparse_layer_start=2, correction_layer=9
):
    def wrap_forward(module):

        def new_tidal_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            top_k=top_k,
            sparse_layer_start=sparse_layer_start,
            correction_layer=correction_layer,
            **kwargs,
        ):
            return llama_tidal_attention_forward(
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
                sparse_layer_start=sparse_layer_start,
                correction_layer=correction_layer,
                **kwargs,
            )

        module.flash_forward = module.forward
        if attn_type == "tidal":
            module.forward = new_tidal_forward


    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_tidal_attention(
                module, top_k, attn_type, sparse_layer_start, correction_layer
            )
        global layer_id
        base_class = type(module).__bases__[0]
        if base_class.__module__ == "src.models.llama_tidaldecoding" and base_class.__name__ == "LlamaAttention":
            wrap_forward(module)
        elif module.__class__.__name__ == "LlamaAttention":
            # For yarn model
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                llama_tidal_forward_yarn, model._modules[name]
            )

            model._modules[name].token_budget = top_k
            model._modules[name].sparse_layer_start = sparse_layer_start
            model._modules[name].correction_layer = correction_layer

