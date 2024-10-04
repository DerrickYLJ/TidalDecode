/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
  This file is modified based on URL:
      https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/handler.cuh
  Support for Page-Sparsity Self-Attention by dynamic selection.
*/

#ifndef FLASHINFER_HANDLER_CUH_
#define FLASHINFER_HANDLER_CUH_

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "flashinfer/rope.cuh"
#include "flashinfer/utils.cuh"

#include "decode/decode_attn.cuh"

namespace flashinfer
{

class BatchDecodeHandler {
public:
	template <typename DType>
	DType* GetTempFloatBuffer() const {
		return (DType*)float_buffer_;
	}
	template <typename IdType>
	IdType* GetNewIndPtr() const {
		return (IdType*)int_buffer_; // + batch_size_after_partition_ + 1;
	}
	template <typename IdType>
	IdType* GetChunkIndPtr() const {
		if(int_buffer_ != nullptr) {
			return ((IdType*)int_buffer_) + batch_size_after_partition_ +
				   1; // batch_size_before_partition_ + 1
		} else {
			return nullptr;
		}
	}
	template <typename IdType>
	IdType* GetBatchIdxMap() const {
		if(int_buffer_ != nullptr) {
			return ((IdType*)int_buffer_) + batch_size_after_partition_ +
				   batch_size_before_partition_ + 2; // batch_size_after_partition_
		} else {
			return nullptr;
		}
	}

	template <PageStorage page_storage,
			  QKVLayout kv_layout,
			  typename DTypeIn,
			  typename DTypeOut,
			  typename IdType>
	cudaError_t BeginForward(IdType* indptr,
							 uint32_t batch_size,
							 uint32_t num_qo_heads,
							 uint32_t num_kv_heads,
							 uint32_t head_dim,
							 uint32_t page_size,
							 RotaryMode rotary_mode) {
		// TODO: Support more general cases.
		static_assert(page_storage == PageStorage::kIndices,
					  "Only support PageStorage::kIndices for now.");
		assert(num_qo_heads == num_kv_heads); // Need to modify page layout for general cases.
		batch_size_before_partition_ = batch_size;
		uint32_t tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size;
		auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimation<page_storage,
																			  kv_layout,
																			  DTypeIn,
																			  DTypeOut,
																			  IdType>;
		FLASHINFER_CUDA_CALL(work_estimation_func(tmp_size,
												  max_grid_size,
												  max_num_pages_per_batch,
												  new_batch_size,
												  batch_size,
												  indptr,
												  num_qo_heads,
												  num_kv_heads,
												  head_dim,
												  page_size,
												  rotary_mode,
												  stream_));
		batch_size_after_partition_ = new_batch_size;
		if(tmp_size > 0) {
			FLASHINFER_CUDA_CALL(cudaMallocAsync(&float_buffer_, tmp_size, stream_));
			FLASHINFER_CUDA_CALL(cudaMallocAsync(
				&int_buffer_,
				sizeof(IdType) * (2 * new_batch_size + batch_size_before_partition_ + 2),
				stream_));
			FLASHINFER_CUDA_CALL(PartitionPagedKVCacheComputeAuxiliaryInfo(max_num_pages_per_batch,
																		   batch_size,
																		   page_size,
																		   indptr,
																		   GetNewIndPtr<IdType>(),
																		   GetChunkIndPtr<IdType>(),
																		   GetBatchIdxMap<IdType>(),
																		   stream_));
		}
		forward_started_ = true;
		return cudaSuccess;
	}

	cudaError_t EndForward() {
		forward_started_ = false;
		batch_size_before_partition_ = 0;
		batch_size_after_partition_ = 0;
		if(float_buffer_ != nullptr) {
			FLASHINFER_CUDA_CALL(cudaFreeAsync(float_buffer_, stream_));
			float_buffer_ = nullptr;
		}
		if(int_buffer_ != nullptr) {
			FLASHINFER_CUDA_CALL(cudaFreeAsync(int_buffer_, stream_));
			int_buffer_ = nullptr;
		}
		return cudaSuccess;
	}

	bool IsForwardStarted() const {
		return forward_started_;
	}

	uint32_t GetBatchSizeBeforePartition() const {
		return batch_size_before_partition_;
	}

	uint32_t GetBatchSizeAfterPartition() const {
		return batch_size_after_partition_;
	}

	cudaStream_t GetCUDAStream() const {
		return stream_;
	}

	void SetCUDAStream(cudaStream_t stream) {
		stream_ = stream;
	}

	BatchDecodeHandler()
		: batch_size_after_partition_(0U)
		, float_buffer_(nullptr)
		, int_buffer_(nullptr)
		, forward_started_(false)
		, stream_(nullptr) { }
	~BatchDecodeHandler() {
		EndForward();
	}

private:
	uint32_t batch_size_before_partition_;
	uint32_t batch_size_after_partition_;
	void* float_buffer_;
	void* int_buffer_;
	bool forward_started_;
	cudaStream_t stream_;
};

/*!
 * \brief Wrapper of BatchDecodeWithPagedKVCache function, and caches the temporary buffer
 *   for cooperative kernels.
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam kv_layout The layout of last 3 dimensions in KV-Cache
 * \tparam DTypeIn The data type of input tensor.
 * \tparam DTypeOut The data type of output tensor.
 * \tparam IdType The data type of index tensor.
 * \param handler The handler for the batch decode forward request.
 * \param q The input tensor.
 * \param paged_kv The paged key-value tensor.
 * \param o The output tensor.
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of heads.
 * \param rotary_mode The rotary mode.
 * \param qk_product The product output of Q and K.
 * \param rope_scale The scale of rope.
 * \param rope_theta The theta of rope.
 * \param stream The CUDA stream.
 * \note This wrapper function should be only called after we call BeginForward function in the
 *   BatchDecodeHandler.
 */
template <PageStorage page_storage,
		  QKVLayout kv_layout,
		  typename DTypeIn,
		  typename DTypeOut,
		  typename IdType>
cudaError_t
BatchDecodeWithPagedKVCacheWrapper(BatchDecodeHandler* handler,
								   DTypeIn* q,
								   paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
								   DTypeOut* o,
								   float* lse,
								   uint32_t num_qo_heads,
								   DTypeOut* qk_product = nullptr,
								   RotaryMode rotary_mode = RotaryMode::kNone,
								   float rope_scale = 1.f,
								   float rope_theta = 1e4,
								   cudaStream_t stream = nullptr) {
	paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> new_paged_kv = paged_kv;
	kv_partition_info_t<IdType> kv_partition_info;
	DTypeOut* tmp = handler->GetTempFloatBuffer<DTypeOut>();
	if(handler->IsForwardStarted()) {
		if(tmp != nullptr) {
			// create auxiliary information for cooperative kernels
			new_paged_kv.batch_size = handler->GetBatchSizeAfterPartition();
			new_paged_kv.indptr = handler->GetNewIndPtr<IdType>();
			kv_partition_info.batch_size_before_partition = handler->GetBatchSizeBeforePartition();
			kv_partition_info.chunk_indptr = handler->GetChunkIndPtr<IdType>();
			kv_partition_info.batch_idx_map = handler->GetBatchIdxMap<IdType>();
		}
	} else {
		std::ostringstream err_msg;
		err_msg << "Please call BatchDecodeHandler's BeginForward() before calling "
				   "BatchDecodeWithPagedKVCacheWrapper()";
		throw std::runtime_error(err_msg.str());
	}
	return BatchDecodeWithPagedKVCache<page_storage, kv_layout, DTypeIn, DTypeOut, IdType>(
		q,
		new_paged_kv,
		kv_partition_info,
		o,
		tmp,
		lse,
		num_qo_heads,
		qk_product,
		rotary_mode,
		rope_scale,
		rope_theta,
		stream);
}

} // namespace flashinfer
#endif // FLASHINFER_HANDLER_CUH_
