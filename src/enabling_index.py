def enable_src(model, top_k):
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from src.index_build.modify_llama import (
            enable_llama_index_build_attention,
        )

        enable_llama_index_build_attention(model, top_k)
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from src.index_build.modify_gpt_neox import (
            enable_gpt_neox_index_build_attention,
        )

        enable_gpt_neox_index_build_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from src.index_build.modify_falcon import (
            enable_falcon_index_build_attention,
        )

        enable_falcon_index_build_attention(model)
    else:
        raise ValueError(f"got {model.config.model_type}")
    return
