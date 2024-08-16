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
import faiss
import numpy as np
from mpi4py import MPI

class KeyValueIndexStore:
    def __init__(self, res, dimension, num_kv_heads, top_k, prefilling, layer_idx, device_id):
        self.dimension = dimension
        self.num_kv_heads = num_kv_heads
        self.key_index_store = [
            faiss.IndexFlatL2(dimension) for _ in range(num_kv_heads)  # on CPU
        ]
        self.key_index_store = [
            faiss.index_cpu_to_gpu(res, device_id, index) for index in self.key_index_store
        ]
        self.top_k = top_k
        self.past_key_value = prefilling # distributedly stored
        self.layer_idx = layer_idx
    
    def update_kv_cache(self, key_states, value_states):
        self.past_key_value[0] = torch.cat([self.past_key_value[0], key_states], dim=2)
        self.past_key_value[1] = torch.cat([self.past_key_value[1], value_states], dim=2)

    def update_index_store(self, key_states):
        bsz, partial_kv_heads, q_len, head_dim = key_states.shape
        if partial_kv_heads != self.num_kv_heads:
            raise ValueError(
                f"dimension of key_states when updating index store is wrong: should be {self.num_kv_heads} but got {partial_kv_heads}"
            )
        for head in range(self.num_kv_heads):
            keys = key_states[:, head, :, :].reshape(-1, head_dim)
            keys = np.ascontiguousarray(keys).astype(np.float32)
            self.key_index_store[head].add(keys)

    def batch_search_index_store(self, query_states):
        # query_states is now a numpy array of float32
        if self.top_k is None:
            raise ValueError("Top-k is None!")
            
        bsz, num_heads, q_len, head_dim = query_states.shape
        self.kv_group_num = num_heads // self.num_kv_heads
        retrieved_k = []
        retrieved_v = []
        
        if q_len > 1:
            raise ValueError("Index Retrieval only supports generation!")
        
        # Perform batched retrieval for each key-value head
        for i_index in range(self.num_kv_heads):
            # Gather queries for the current key-value head
            head_indices = np.arange(i_index * self.kv_group_num, (i_index + 1) * self.kv_group_num)
            queries = query_states[:, head_indices, :, :].reshape(-1, head_dim)
            queries = np.ascontiguousarray(queries).astype(np.float32)
            
            # Perform the batched search on the current index store
            _, I_k = self.key_index_store[i_index].search(queries, k=self.top_k)
            
            # Sort the indices and retrieve keys and values
            sorted_indices = np.argsort(I_k, axis=1)
            sorted_I_k = np.take_along_axis(I_k, sorted_indices, axis=1)
            
            # Retrieve keys and values from past_key_value
            keys_retrieved = self.past_key_value[0][:, i_index, :, :].reshape(-1, head_dim)[sorted_I_k].reshape(bsz, -1, self.top_k, head_dim)
            values_retrieved = self.past_key_value[1][:, i_index, :, :].reshape(-1, head_dim)[sorted_I_k].reshape(bsz, -1, self.top_k, head_dim)
            
            retrieved_k.append(keys_retrieved)
            retrieved_v.append(values_retrieved)
        
        retrieved_k = np.concatenate(retrieved_k, axis=1)
        retrieved_v = np.concatenate(retrieved_v, axis=1)

        # Validate the shapes of retrieved_k and retrieved_v
        if retrieved_k.shape != (bsz, num_heads, self.top_k, head_dim):
            raise ValueError(
                f"retrieved shape is incorrect, should be ({bsz, num_heads, self.top_k, head_dim}), but got {retrieved_k.shape}"
            )
        if retrieved_k.shape != retrieved_v.shape:
            raise ValueError(
                f"retrieved_k and retrieved_v are mismatched, retrieved_k: {retrieved_k.shape} but retrieved_v: {retrieved_v.shape}"
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
    comm = None,
    total_rank = 1,
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
    
    # Timing start for data transfer
    start_transfer_time = time.time()
    key_states_cpu = key_states.to("cpu")
    value_states_cpu = value_states.to("cpu")
    end_transfer_time = time.time()
    transfer_time = end_transfer_time - start_transfer_time

    # key_states_cpu and value_states_cpu are torch.tensor of fp16
    key_states_cpu_np = np.ascontiguousarray(key_states_cpu.numpy())
    value_states_cpu_np = np.ascontiguousarray(value_states_cpu.numpy())
    key_states_cpu_np = key_states_cpu_np.astype(np.float32)
    value_states_cpu_np = value_states_cpu_np.astype(np.float32)
    key_states_cpu_splited = np.array_split(key_states_cpu_np, total_rank, axis=1)
    value_states_cpu_splited = np.array_split(value_states_cpu_np, total_rank, axis=1)
    key_states_cpu_splited = np.stack(key_states_cpu_splited, axis=0)
    value_states_cpu_splited = np.stack(value_states_cpu_splited, axis=0)
    key_recvbuf = np.empty((bsz, self.num_key_value_heads//total_rank, kv_seq_len, self.head_dim), dtype=np.float32)
    value_recvbuf = np.empty((bsz, self.num_key_value_heads//total_rank, kv_seq_len, self.head_dim), dtype=np.float32)
    
    if past_key_value is not None:
        comm.Scatter(key_states_cpu_splited, key_recvbuf, root=0)
        comm.Scatter(value_states_cpu_splited, value_recvbuf, root=0)
        start_kv_update_time = time.time()
        self.kv_index_store.update_kv_cache(torch.tensor(key_recvbuf), torch.tensor(value_recvbuf))
        end_kv_update_time = time.time()
        kv_update_time = end_kv_update_time - start_kv_update_time
        if key_states.size() != value_states.size():
            raise ValueError(
                f"key_states and value_states are mismatched, key_states: {key_states.size()} but value_states: {value_states.size()}"
            )
    else:
        if self.layer_idx == 0:
            index_store_config = [self.head_dim, self.num_key_value_heads, top_k, self.layer_idx, bsz, self.num_key_value_heads//total_rank, q_len, self.head_dim, self.num_heads]
            print(f"pid: {comm.Get_rank()}; index_store_config: {index_store_config}")
            index_store_config = comm.bcast(index_store_config, root=0)
        comm.Scatter(key_states_cpu_splited, key_recvbuf, root=0)
        comm.Scatter(value_states_cpu_splited, value_recvbuf, root=0)
        self.kv_index_store = KeyValueIndexStore(
            self.res, self.head_dim, self.num_key_value_heads//total_rank, top_k, [torch.tensor(key_recvbuf), torch.tensor(value_recvbuf)], self.layer_idx, 0
        )
        past_key_value = (key_states_cpu, value_states_cpu)

    # Timing start for index update
    start_index_update_time = time.time()
    self.kv_index_store.update_index_store(key_recvbuf)
    end_index_update_time = time.time()
    index_update_time = end_index_update_time - start_index_update_time

    if top_k is not None and q_len == 1:
        # Timing start for index search
        start_index_search_time = time.time()
        query_states_cpu = query_states.to("cpu") #  (bsz, self.num_heads, q_len, self.head_dim)
        query_states_cpu_np = np.ascontiguousarray(query_states_cpu.numpy())
        query_states_cpu_np = query_states_cpu_np.astype(np.float32)
        query_states_cpu_splited = np.array_split(query_states_cpu_np, total_rank, axis=1)
        query_states_cpu_splited = np.stack(query_states_cpu_splited, axis=0)
        query_recvbuf = np.empty((bsz, self.num_heads//total_rank, q_len, self.head_dim), dtype=np.float32)
        comm.Scatter(query_states_cpu_splited, query_recvbuf, root=0)
        key_states, value_states = self.kv_index_store.batch_search_index_store(query_recvbuf)
        key_gatherbuf = np.empty((bsz, self.num_heads, top_k, self.head_dim), dtype=np.float32)
        value_gatherbuf = np.empty((bsz, self.num_heads, top_k, self.head_dim), dtype=np.float32)
        comm.Gather(key_states, key_gatherbuf, root=0)
        comm.Gather(value_states, value_gatherbuf, root=0)
        key_states = torch.tensor(key_gatherbuf, dtype=torch.float16)
        value_states = torch.tensor(value_gatherbuf, dtype=torch.float16)
        end_index_search_time = time.time()
        index_search_time = end_index_search_time - start_index_search_time

        start_transfer_time = time.time()
        key_states = key_states.to(hidden_states.device)
        value_states = value_states.to(hidden_states.device)
        end_transfer_time = time.time()
        transfer_time += end_transfer_time - start_transfer_time
        print(f"0: {key_states.shape}")
    else:
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

    # Print out timing results
    print(f"pid: {comm.Get_rank()}")
    print(f"Data Transfer Time: {transfer_time:.6f} seconds")
    print(f"Index Update Time: {index_update_time:.6f} seconds")
    print(f"Index Search Time: {index_search_time:.6f} seconds" if top_k is not None and q_len == 1 else "Index Search Time: 0.0 seconds")
    print(f"KV cache Update Time: {kv_update_time:.6f} seconds" if top_k is not None and q_len == 1 else "KV cache Update Time: 0.0 seconds")
    print(torch.cuda.memory_summary())
    return attn_output, attn_weights, past_key_value


def enable_llama_index_build_attention(model, top_k, comm=None):
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
            comm=comm,
        ):
            return llama_index_build_attention_forward(
                module,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                top_k=top_k,
                comm=comm,
                total_rank=comm.Get_size(),
            )

        module.forward = new_forward
    if model != None and comm!=None:
        if comm.Get_rank() == 0:
            print(f"pid: {comm.Get_rank()}")
            for name, module in reversed(model._modules.items()):
                if len(list(module.children())) > 0:
                    enable_llama_index_build_attention(module, top_k, comm)
                if isinstance(module, LlamaAttention):
                    wrap_forward(module)
    else:
        if comm != None:
            pid = comm.Get_rank()
            total_rank = comm.Get_size()
            print(f"pid: {pid}, start communication")
            
            store_list = []
            res = faiss.StandardGpuResources() # gpu-index resource
            index_store_config = [] # configs for index store: 
            index_store_config =  comm.bcast(index_store_config, root=0)
            [head_dim, num_key_value_heads, top_k, layer_idx, bsz, group_size, q_len, head_dim, num_heads] = index_store_config
            print(f"pid: {pid}; index_store_config: {index_store_config}")

            # initialize index stores
            while layer_idx < 32:
                key_recvbuf = np.empty((bsz, group_size, q_len, head_dim), dtype=np.float32)
                value_recvbuf = np.empty((bsz, group_size, q_len, head_dim), dtype=np.float32)
                # Scatter the key and value states to each process
                comm.Scatter(None, key_recvbuf, root=0)
                comm.Scatter(None, value_recvbuf, root=0)
                store_list.append(KeyValueIndexStore(res, head_dim, group_size, top_k, [torch.tensor(key_recvbuf), torch.tensor(value_recvbuf)], layer_idx, pid))
                layer_idx += 1
                # Update the index store
                store_list[-1].update_index_store(key_recvbuf)
            
            # update kv_cache + index_store + search 
            # until main process sends abort signal
            while True:
                for i in range (32):
                    kv_seq_len = store_list[i].past_key_value[0].shape[-2]
                    query_recvbuf = np.empty((bsz, num_heads//total_rank, 1, head_dim), dtype=np.float32)
                    key_recvbuf = np.empty((bsz, group_size, 1, head_dim), dtype=np.float32)
                    value_recvbuf = np.empty((bsz, group_size, 1, head_dim), dtype=np.float32)
                    comm.Scatter(None, key_recvbuf, root=0)
                    comm.Scatter(None, value_recvbuf, root=0)
                    # update kv_cache
                    store_list[i].update_kv_cache(torch.tensor(key_recvbuf), torch.tensor(value_recvbuf))
                    # update index_store
                    store_list[i].update_index_store(key_recvbuf)
                    # batch search
                    comm.Scatter(None, query_recvbuf, root=0)
                    key_states, value_states = store_list[i].batch_search_index_store(query_recvbuf)
                    print(f"{pid}: {key_states.shape}")
                    comm.Gather(key_states, None, root=0)
                    comm.Gather(value_states, None, root=0)

