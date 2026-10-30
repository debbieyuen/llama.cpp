# Kimi K3 A10 benchmark suite

Built for the debbie-kimi-k3 llama.cpp fork, using llama-server's native HTTP API.
The Anyscale spreadsheet informed the metrics and workload coverage. This is not
an Anyscale deployment manager or a vLLM/OpenAI-chat client. Raw completion prompts
and different hardware/models are not directly comparable to the sheet's GLM-5.2 results.

## Install and transfer

Upload this whole directory with Jupyter's file browser into ~/llama.cpp/a10-benchmark.
In the existing virtual environment:

```bash
cd ~/llama.cpp
source .venv/bin/activate
python -m pip install -r a10-benchmark/requirements.txt
```

## Start the model

```bash
python a10-benchmark/launch.py --context-per-slot 2048
```

Run in one terminal; benchmarks run in another. The launcher binds localhost,
records the exact Git revision, command and controlled environment, saves a raw
server log, and samples GPU utilization/memory/power, host memory, and process I/O/RSS
every few seconds. Ctrl+C terminates only the launched server process group.
GPU statistics are device-wide, not attributable exclusively to this server.
Linux /proc data and nvidia-smi are optional; unavailable telemetry is omitted.

Defaults are an EXPERIMENTAL starting point, not a proven Kimi K3 fit:
100 GiB host expert cache, 20 GPU layers, CPU experts, 16 CPU threads,
2048 context tokens in one slot. Loading and correctness must pass first.
Long contexts or more parallel slots can require substantially more memory.

Change one variable at a time, stop the server between configurations:

```bash
python a10-benchmark/launch.py --cache-mib 81920 --gpu-layers 20 --threads 16
python a10-benchmark/launch.py --cache-mib 122880 --gpu-layers 20 --threads 16
python a10-benchmark/launch.py --cache-mib 102400 --gpu-layers 30 --threads 16
python a10-benchmark/launch.py --cache-mib 102400 --gpu-layers 20 --threads 8
python a10-benchmark/launch.py --reader pread --second-chance 3
```

Use --dry-run to inspect a launch without starting it. Cache sizes are MiB.
The launcher does not force PCIe host-bank execution, change quantization,
turn on MTP, or modify the number of experts selected by the model.
Retain raw logs for cache-hit rates, read throughput, and backend evidence.

## Full suite, selectable workloads

In a second terminal, activate the same virtual environment. Run from ~/llama.cpp.
`--context-per-slot` must match the server's actual per-slot capacity. Cases that
exceed it are explicitly recorded as skipped. Do not call a skipped test a pass.

### Smoke correctness

```bash
bash a10-benchmark/run.sh --suite smoke --context-per-slot 2048 --output 64 --requests 1 --warmups 0
```

Checks arithmetic, exact JSON formatting, field extraction, and instruction following. This is a diagnostic check, not a quality benchmark.
Correctness uses an exact stripped-text match: extra prose or reasoning fails the
format check. Inspect saved answers to distinguish formatting from factual failure.

### Prompt and output lengths

```bash
bash a10-benchmark/run.sh --suite performance --context-per-slot 2048 \
  --input 64 256 1024 --output 32 128 512 --concurrency 1 --requests 5 --warmups 1
```

Full workload matrix (requires a server actually configured with adequate context):

```bash
bash a10-benchmark/run.sh --suite performance --context-per-slot 131072 \
  --input 64 1024 4096 8192 32768 65536 98304 130000 \
  --output 256 512 2048 --concurrency 1 4 8 16 32 --requests 200 --warmups 5
```

This can take an extremely long time on an A10 with SSD-streamed Kimi K3.
The full matrix is available but not run automatically. Output length is forced
with ignore_eos for performance/repeatability/stability, and is a maximum for
correctness tests. Record actual output length. Repeated synthetic archive text
is controlled workload data, not representative natural language quality data.

### Retrieval and context position

```bash
bash a10-benchmark/run.sh --suite retrieval --context-per-slot 2048 \
  --input 256 1024 1900 --output 64 --positions 0.1 0.5 0.9 --requests 3
```

Increase input lengths through the full matrix above after establishing memory fit.
The random-looking fixed code is inserted at the specified fraction of filler
TOKENS (not the entire formatted prompt). Prompts are token arrays of exactly the
requested size, excluding server-added special tokens. The actual server prompt
count is also recorded. Each prompt token array and expected answer is saved.
For boundary testing use input=context-output, then inspect actual server token
counts and any rejection; the server may add a BOS token. No context shifting is
allowed by the launcher. Truncation reported by the server is a test failure.

### Repeatability

```bash
bash a10-benchmark/run.sh --suite repeatability --context-per-slot 2048 \
  --input 64 1024 --output 128 --concurrency 1 --requests 5 --warmups 1
```

`unique_texts` should be 1 for each identical-prompt case if output is repeatable.
Different text hashes are evidence of a difference, not automatically a model bug.
To compare cache/GPU/thread configurations, use matching prompt and output settings
and compare the saved text and token IDs. GPU vs CPU arithmetic can differ.

### Concurrency and sustained load

Start a server with --parallel N and enough context per slot before interpreting
concurrency as simultaneous model execution. Higher client concurrency than server
slots measures queueing too. Requests use a closed-loop worker pool, no retries.

```bash
bash a10-benchmark/run.sh --suite performance --context-per-slot 2048 \
  --input 64 --output 256 --concurrency 1 2 4 8 --requests 200 --warmups 5
bash a10-benchmark/run.sh --suite stability --context-per-slot 2048 \
  --input 1024 --output 128 --concurrency 1 --duration 1800 --warmups 1
```

Duration is a request-admission window PER CASE; in-flight requests are allowed to
finish up to the per-request timeout. Errors, empty streams, truncated streams,
truncation flags, timeouts, and HTTP failures are recorded rather than retried.
Repeated runs use new timestamped directories and do not overwrite previous results.

## Metrics and evidence

- TTFT: client request start to first nonempty text/token event, includes queue/network.
- TPOT: first to last received token event divided by actual output token count minus
  one; null for one token. Transport buffering makes this a client approximation,
  not exact GPU token latency. Chunk counts are never treated as token counts.
- E2E: request start to end of stream. p50/p95/p99 computed over successful requests.
- Output throughput: sum of actual successful output tokens / measured phase wall time,
  including worker scheduling and failures, excluding warmups and metrics snapshots.
- Server prompt/decode speeds are separate from client latency and aggregate throughput.
- Exact-match correctness, failures and unique response hashes are separate measures.
- /metrics snapshots bracket only the measured phase. They are global server counters;
  other clients can contaminate deltas. Warmups are stored separately.
- cache_prompt is FALSE by default. --cache-prompt explicitly enables it. This does
  not flush OS caches or the expert cache: warm/cold storage state is not controlled.
- Warmups prime the server; no claim is made about cold-start latency. Startup evidence
  is in the launcher logs. No automatic system-wide cache clearing occurs.
- Small samples cannot establish stable p99 estimates. Use adequate repetitions.

Each run saves manifest.json, summary.json, summary.csv, server properties when
available, per-case prompts, per-request JSON including text and raw final timing
metadata, requests.jsonl, and metrics snapshots. CSV can be imported into Sheets.
No data is written to your existing spreadsheet. API keys come only from LLAMA_API_KEY
(or --api-key-env), are not saved, and the client uses normal TLS verification.

The native API can be accessed via a tunnel using --url. A public endpoint must be
configured separately; the launcher intentionally exposes only localhost. Ray/vLLM,
MTP acceptance, interconnect, and attention backend must be established from their
own deployment logs, not inferred from these client results.

## Tests

```bash
python -m unittest discover -s a10-benchmark -p 'test_*.py' -v
```

Tests use a fake HTTP transport and do not load a model or consume GPU resources.

## Run all test modes

```bash
CONTEXT_PER_SLOT=2048 bash a10-benchmark/full-suite.sh
```

This runs smoke, retrieval, repeatability, the full performance matrix, and stability,
in that order, stopping on a failed suite. Context-incompatible cases are skipped;
2048 does not claim long-context coverage. Change CONTEXT_PER_SLOT only after launching
and verifying a server with that capacity. BENCH_URL optionally selects a different URL.
Full performance runs use 200 requests per case and five warmups. Choose individual
suites for shorter experiments; the full sequence is deliberately substantial.

Reference requirements: https://docs.google.com/spreadsheets/d/1ypENMBfY8rBuph0EWbyrNJPxsLaPStk0Xn20IjYJnGw/edit?gid=695739133
The spreadsheet was read as project context; its existing results were not modified.
