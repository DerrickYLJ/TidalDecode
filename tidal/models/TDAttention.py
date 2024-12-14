import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

import tidal.utils


class TDAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, att_type="full"):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings
        self.att_type = att_type
        self.rope_type = None
        self.rope_config = None

        assert self.att_type in ["full", "search", "sparse"], f"Unknown attention type {att_type}, should be in [full, search, sparse]"

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        # rope_theta is default to 1e4, as set in RoPE kernel API.
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.rope_theta = self.config.rope_theta

        rope_scaling = getattr(self.config, "rope_scaling", None)
        if rope_scaling is None:
            self.rope_scale = 1.0
            self.rope_type = "llama2"
        else:
            if rope_scaling is not None:
                if "type" in rope_scaling:
                    rope_type = rope_scaling["type"]
                elif "rope_type" in rope_scaling:
                    rope_type = rope_scaling["rope_type"]
                else:
                    raise ValueError(
                        "rope_scaling must have a 'type' or 'rope_type' key.")
                assert rope_type in ["llama3"]
                self.rope_type = rope_type
                self.rope_config = rope_scaling
                self.rope_scale = rope_scaling["factor"]
                self.rope_theta = getattr(self.config, "rope_theta", None)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        iController: Optional[tidal.utils.InferenceController] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        assert bsz == 1, "TDAttention only supports batch size 1."
        assert hasattr(
            self, "layer_idx"
        ), "TDAttention requires layer_idx to inference."

        if self.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            torch.cuda.nvtx.range_push("qkv_proj")
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            torch.cuda.nvtx.range_pop()

        # Not transposed for Append kv cache NHD layout
        # if q_len == 1 and self.att_type == "sparse":
        #     print(self.layer_idx, hidden_states[0, 0, :5])
        if True:
            # Test RoPE
            query_states = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            if iController.kv_cache.seqlen - q_len > 0:
                position_ids[0, 0] = iController.kv_cache.seqlen - q_len

            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            query_states = query_states.transpose(1, 2).view(q_len, self.num_heads, self.head_dim)
            key_states = key_states.transpose(1, 2).view(q_len, self.num_key_value_heads, self.head_dim)
            value_states = value_states.transpose(1, 2).view(q_len, self.num_key_value_heads, self.head_dim)
        else:
            query_states = query_states.view(q_len, self.num_heads, self.head_dim)
            key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
            value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)

            torch.cuda.nvtx.range_push("RoPE")
            if self.rope_type == "llama2":
                tidal.utils.apply_rope_in_place(
                    query_states,
                    key_states,
                    iController.kv_cache.seqlen - q_len,
                    rope_scale=self.rope_scale,
                )
            elif self.rope_type == "llama3":
                tidal.utils.apply_llama31_rope_in_place(
                    query_states,
                    key_states,
                    iController.kv_cache.seqlen - q_len,
                    rope_scale=self.rope_scale,
                    rope_theta=self.rope_theta,
                    low_freq_factor=self.rope_config["low_freq_factor"],
                    high_freq_factor=self.rope_config["high_freq_factor"],
                    old_context_length=self.rope_config["original_max_position_embeddings"]
                )
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("append_kv")
        # Tidal manages KV-Cache internal (with PageAttention)
        # Here we do not concat / stack
        # We concat after RoPE
        tidal.utils.append_kv(
            key_states,
            value_states,
            iController,
            self.layer_idx,
        )
        torch.cuda.nvtx.range_pop()

        ## You are here

        # Prefill/Decode kernels is different
        if q_len > 1:
            torch.cuda.nvtx.range_push("prefill_attn")
            torch.cuda.synchronize()
            attn_output = tidal.utils.prefill_forward(
                query_states,
                iController,
                self.layer_idx,
            )
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        else:
            # Decode Stage
            # Skipping layers is controled by PAGE_BUDGET, which is set in LlamaModel.
            if self.att_type == "full":
                torch.cuda.nvtx.range_push("full_attn")
                torch.cuda.synchronize()
                # print(query_states[0, 0, :8], key_states[0, 0, :8], value_states[0, 0, :8])
                attn_output = tidal.utils.decode_sparse_attn(
                    query_states,
                    iController,
                    self.layer_idx,
                    iController.kv_indices_without_last,
                    False,
                )
                # print(iController.kv_indices_without_last)
                # print(f"td-layer-{self.layer_idx}, attn_output: {attn_output[0, 0, 0]}")
                # exit(0)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            elif self.att_type == "search":
                torch.cuda.nvtx.range_push("search_attn")
                torch.cuda.synchronize()

                attn_output = tidal.utils.decode_sparse_attn(
                    query_states,
                    iController,
                    self.layer_idx,
                    iController.kv_indices_without_last,
                    True,
                )
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("arg_topk")
                torch.cuda.synchronize()
                # tidal.utils.decode_topk(
                #     iController,
                # )
                # print(iController.td_token_budget)
                token_budget = min(iController.td_token_budget-1, iController.kv_indices_without_last.shape[1])
                _, top_k_indices = torch.topk(iController.qk_product[:, :iController.kv_indices_without_last.shape[1]], k=token_budget, dim=-1)
                # top_k_indices, _ = torch.sort(top_k_indices, dim=-1)
                iController.top_k_indices = top_k_indices.int()
                # if self.layer_idx == 13:
                #     print(f"td-layer-{self.layer_idx}, qk_product: {iController.qk_product[0, 40:46]/1.44336}")
                # if self.layer_idx == 2:
                #     print(iController.kv_indices_without_last.shape, iController.top_k_indices.shape)
                #     # print(self.layer_idx, iController.qk_product[0, :61]/1.44336, query_states[0, 0, :20], key_states[0, 0, :20])
                # exit(0)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                # iController.top_k_indices = torch.ones((32, 32)).to(attn_output.device).int()
                # print(f"td-layer-{self.layer_idx}, top_k_indices: {iController.top_k_indices[0]}")
                
            elif self.att_type == "sparse":
                # print(self.layer_idx, iController.top_k_indices)
                assert iController.top_k_indices is not None, "top_k_indices should be set in search stage."
                torch.cuda.nvtx.range_push("sparse_attn")
                torch.cuda.synchronize()
                attn_output = tidal.utils.decode_sparse_attn(
                    query_states,
                    iController,
                    self.layer_idx,
                    iController.top_k_indices,
                    False,
                )
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()

        attn_output = attn_output.unsqueeze(0)  # unsqueeze the batch dimension
        # FlashInfer output is naturally NHD
        # Note that we manully control NHD. Should be more general
        if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        torch.cuda.nvtx.range_push("o_proj")
        if self.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop()

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
