#include "bsk_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

// Note that estimated_indices does not contain the last page
void topk_filtering(torch::Tensor input_value,
							 torch::Tensor input_indices,
							 torch::Tensor d_out,
							 torch::Tensor indices_out,
							 torch::Tensor buf,
							 unsigned int token_budget) {
	#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(input_value); // [num_heads, num_pages]
	CHECK_INPUT(input_indices); // [num_heads, num_pages]
	CHECK_DIM(2, input_value);
	CHECK_DIM(2, input_indices);
	#endif

	auto num_heads = input_value.size(0);
	auto kv_len = input_value.size(1);

	#ifdef BSK_TORCH_CHECK
	CHECK_EQ(num_heads, input_indices.size(0));
	CHECK_EQ(input_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(32, num_heads); // Not necessary, but for Llama-7b
	CHECK_EQ(token_budget, d_out.size(1));
	CHECK_EQ(token_budget, indices_out.size(1));
	#endif

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(input_value.scalar_type(), c_type, [&] {
		decode_select_k<c_type, int32_t, 32>(
			static_cast<c_type*>(input_value.data_ptr()),
			static_cast<int32_t*>(input_indices.data_ptr()),
			static_cast<char*>(buf.data_ptr()),
			kv_len,
			token_budget,
			static_cast<c_type*>(d_out.data_ptr()),
			static_cast<int32_t*>(indices_out.data_ptr()),
			true);
		return true;
	});
	TORCH_CHECK(success, "Top-k filtering failed to dispatch with dtype ", input_value.scalar_type());
}