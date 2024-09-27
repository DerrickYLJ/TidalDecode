from mpi4py import MPI


def enable_src(model, top_k, comm, attn_type, sparse_layer_start, correction_layer):
    # Broadcast model.config.model_type to all ranks
    if comm != None:
        model_type = model.config.model_type if comm.Get_rank() == 0 else None
        model_type = comm.bcast(model_type, root=0)
    else:
        model_type = model.config.model_type
    if "llama" in model_type:
        k_seq_dim = v_seq_dim = 2
        if comm != None:
            from src.index_build.modify_llama_mpi import (
                enable_llama_index_build_attention,
            )
        else:
            from src.index_build.modify_llama import (
                enable_llama_index_build_attention,
            )
        enable_llama_index_build_attention(model, top_k, comm, attn_type, sparse_layer_start, correction_layer)
    elif "mpt" in model_type:
        v_seq_dim = 2
        k_seq_dim = 3
        from src.index_build.modify_mpt import (
            enable_mpt_index_build_attention,
        )

        enable_mpt_index_build_attention(model)
    elif "gpt_neox" in model_type:
        k_seq_dim = v_seq_dim = 2
        from src.index_build.modify_gpt_neox import (
            enable_gpt_neox_index_build_attention,
        )

        enable_gpt_neox_index_build_attention(model)
    elif "falcon" in model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from src.index_build.modify_falcon import (
            enable_falcon_index_build_attention,
        )

        enable_falcon_index_build_attention(model)
    else:
        raise ValueError(f"got {model_type}")
    return
