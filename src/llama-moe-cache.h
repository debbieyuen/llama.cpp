#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // kimi-k3-spark patch 0003: predictive prefetch (LLAMA_EXPERT_STORE_PREFETCH=1, expert store with the slot maps on
    // the GPU and the asynchronous host-bank hook): the graph adds a lookup named "moe_prefetch-<il>" of the experts
    // that layer il's router picks on the previous layer's input, and the cache reads the predicted misses while the
    // GPU still works on the previous layer. A wrong guess only costs a read.
    bool prefetch_enabled() const;

    // kimi-k3-spark patch 0003: self-prefetch (LLAMA_EXPERT_STORE_SELF_PREFETCH=1, same requirements): the graph posts
    // layer il's real selected experts as a "moe_prefetch-<il>" lookup right after the router, then runs the work that
    // does not need the routed experts (shared expert, latent down projection) before the real lookup, so the misses
    // are read while the GPU computes. The real lookup still plans and waits as before: correctness never depends on it.
    bool self_prefetch_enabled() const;
    // kimi-k3-spark patch 0003: split expert GEMV (LLAMA_EXPERT_STORE_SPLIT=1, same requirements, single-token decode):
    // the graph looks the selected experts up twice, early ("moe_early-<il>", the experts already complete in the bank,
    // the rest pointing at a zero record) and for real (the misses, the rest pointing at the zero record), runs the
    // expert matmuls on both and merges the rows (merge map), so the resident experts compute while the misses are
    // read. get_split_map: kind 1 early map, 2 late map, 3 merge map, 4 the zero record's slot as an I32 [1, 1].
    bool split_enabled() const;
    // kind 5: an I32 [n_expert_used] of 0, 1, ...; the late and merge maps are read by position with its first
    // split_late_rows() entries
    ggml_tensor * get_split_map(int32_t il, int64_t n_tokens, int kind) const;
    int split_late_rows() const;
    // the experts of w in the cache with the zero record included (one more slot), for the split matmuls
    ggml_tensor * get_experts_split(const ggml_tensor * w) const;

    // kimi-k3-spark patch 0003: true when the slot-map lookups run on the GPU (LLAMA_EXPERT_STORE_GPU_SLOTS, the
    // asynchronous host-bank hook); false when they run in the CPU split and the fills happen in the CPU graph hook
    bool lookups_on_gpu() const;

    // kimi-k3-spark patch 0003: what the predictive prefetch applies the next layer's router to
    // (LLAMA_EXPERT_STORE_PREFETCH_INPUT): 0 = this layer's router input, 1 = the alternative the model passes
    // (Kimi K3: the un-normed FFN input plus the shared expert's output, under the next layer's FFN norm)
    int prefetch_input() const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // kimi-k3-spark patch 0003: expert store (LLAMA_EXPERT_STORE=<file from tools/repack-experts.py>).
    // The cache lives in host RAM and is filled from the store with O_DIRECT reads; small batches are served
    // through the slot maps (filled from the CPU graph hook), large batches are read into pinned staging
    // and uploaded from there by the scheduler copy callback.
    bool fill_prefill(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, const ggml_tensor * ids, const std::vector<bool> & used);
    void prefill_reset();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
