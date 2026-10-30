#!/usr/bin/env bash
# Run every test mode against an already running llama-server.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT="${CONTEXT_PER_SLOT:?Set CONTEXT_PER_SLOT to the running server context per slot}"
URL="${BENCH_URL:-http://127.0.0.1:8080}"
COMMON=(--url "$URL" --context-per-slot "$CONTEXT")
python "$HERE/benchmark.py" "${COMMON[@]}" --suite smoke --output 64 --requests 3
python "$HERE/benchmark.py" "${COMMON[@]}" --suite retrieval --input 256 1024 4096 8192 32768 65536 98304 130000 --output 64 --positions .1 .5 .9 --requests 3
python "$HERE/benchmark.py" "${COMMON[@]}" --suite repeatability --input 64 1024 8192 --output 128 --requests 5
python "$HERE/benchmark.py" "${COMMON[@]}" --suite performance --input 64 1024 4096 8192 32768 65536 98304 130000 --output 256 512 2048 --concurrency 1 4 8 16 32 --requests 200 --warmups 5
python "$HERE/benchmark.py" "${COMMON[@]}" --suite stability --input 1024 --output 512 --concurrency 1 --duration 1800
