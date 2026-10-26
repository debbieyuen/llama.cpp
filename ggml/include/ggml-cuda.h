#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// wrap caller-owned host memory as a "host bank" buffer: registered with the driver and read in place by the
// kernels of an integrated device; NULL if the device cannot do that (also via ggml_backend_reg_get_proc_address)
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_cuda_host_bank_buffer_from_ptr(void * ptr, size_t size);

// called right before a GET_ROWS whose src0 is a host-bank tensor, with that src0 (the MoE cache's slot map, a
// tensor that outlives the graph), the kind of lookup (from the node's name) and its (contiguous, I32) ids read back
// from the device. With the synchronous hook the stream is synchronized first; with the asynchronous one the call
// comes from a poller thread while the GPU waits, possibly after the graph that launched it was rebuilt, which is
// why no graph tensor is passed. For the MoE cache's slot-map lookups: the fills happen here.
#define GGML_CUDA_HOST_BANK_LOOKUP   0 // the real lookup, the GPU waits for the fill
#define GGML_CUDA_HOST_BANK_PREDICT  1 // "moe_prefetch-*": predicted experts, posted without a wait
#define GGML_CUDA_HOST_BANK_SELF     2 // "moe_prefetch_self-*": the layer's real experts posted early, no wait
typedef void (*ggml_backend_cuda_host_bank_hook_t)(void * user_data, const struct ggml_tensor * slot_map, int kind, const int32_t * ids, int64_t n_ids);
GGML_BACKEND_API void ggml_backend_cuda_set_host_bank_hook(ggml_backend_cuda_host_bank_hook_t fn, void * user_data);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
