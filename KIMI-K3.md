# llama.cpp with an SSD-streaming MoE expert store (branch `kimi-k3`)

This branch is upstream llama.cpp at `547208dca250ceb3e4c8615598d043f2b602a527` plus three commits, one per
patch of [xerootg/kimi-k3-spark](https://github.com/xerootg/kimi-k3-spark). That repo is the source of truth:
it holds the patches, the image build and the experiment ledger. This branch makes the same tree browsable and
buildable as a normal llama.cpp checkout. Its tree is byte-identical to what the image build produces by
running `git apply` on `patches/*.patch` in order.

| commit | patch | what it does |
|---|---|---|
| 1 | `0001-prefetch-mmapped-experts` | `madvise(MADV_WILLNEED)` of every selected expert slice for mmap'd MoE experts. On a DGX Spark with Kimi K3 this took decode from 147 s/token to 4.45 s/token. |
| 2 | `0002-zero-copy-host-experts` | Opt-in (`GGML_CUDA_HOST_ZERO_COPY=1`) GPU reads of mmap'd host experts. A dead end on the GB10, where it ran at about 19 s/token; kept off. |
| 3 | `0003-expert-store` | The engine. Routed experts are served from a repacked store file with O_DIRECT / io_uring reads into a host-RAM LRU that the GPU computes from directly. On top of that: self-prefetch and speculative prefetch, split expert GEMV, cache snapshot, memory lock, memory pressure valve, second-chance eviction, and the GH200 extras (HBM tier, prefill lookahead, Hopper Q6_K TMA GEMV). |

Results for Kimi K3 `UD-IQ1_S` (594 GB):

| box | decode |
|---|---|
| one DGX Spark (128 GB unified memory) | 1.93 tok/s deployed, 1.96 tok/s with second-chance eviction |
| a GH200 | about 13-13.8 tok/s |

Every kept change is bit-identical to unpatched llama.cpp at temperature 0.

Read next:

- [docs/improvements.md](https://github.com/xerootg/kimi-k3-spark/blob/main/docs/improvements.md): what each
  change does and what it bought, the dead ends, and how to build, prepare the expert store and run.
- [patches/README.md](https://github.com/xerootg/kimi-k3-spark/blob/main/patches/README.md): every part and every
  environment knob.
- [docs/experiments.md](https://github.com/xerootg/kimi-k3-spark/blob/main/docs/experiments.md): the experiment
  ledger.

The expert store file is built with
[`tools/repack-experts.py`](https://github.com/xerootg/kimi-k3-spark/blob/main/tools/repack-experts.py) from
that repo. The engine is enabled with `LLAMA_EXPERT_STORE=<store file>` plus `--moe-cache-mib <budget>`, with
the experts placed in host memory (`--n-cpu-moe`). Without `LLAMA_EXPERT_STORE` the store is inactive. Two
changes stay on by default even then:

- patch 0001's prefetch for mmap'd experts (`GGML_CPU_MMID_PREFETCH=0` turns it off);
- the Hopper Q6_K GEMV on sm_90 (`GGML_CUDA_MMVQ_Q6K_TMA=0` turns it off).

The branch is not rebased onto newer upstream commits. When the pin in kimi-k3-spark moves, a new branch or tag is
published.
