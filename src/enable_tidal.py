def enable_tidal(model, attn_type="tidal", top_k=256, sparse_layer_start=2, correction_layer=13):   
    model_type = model.config.model_type
    print(f"token budget: {top_k}")
    print(f"sparse layer starts from: Layer {sparse_layer_start}")
    print(f"reselection layer: {correction_layer}")
    if "llama" in model_type:
        # currently support llama family
        from src.tidal_build.modify_llama import (
            enable_llama_tidal_attention,
        )
        enable_llama_tidal_attention(model, top_k, attn_type, sparse_layer_start, correction_layer)
    return