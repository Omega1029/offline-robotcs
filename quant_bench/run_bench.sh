#!/usr/bin/env bash
# Run the edge latency/memory benchmarks on the Jetson. Assumes the venv is active and the
# checkpoint + HF cache are in place (see README_JETSON.md). Synthetic inputs — no simulator.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
PY="${PY:-python3}"

echo "############ 1) rotation overhead (no model, pure torch) ############"
$PY 13_rotation_overhead.py --iters 200 || echo "[warn] rotation overhead failed"

echo "############ 2) bf16 baseline (real edge latency + memory) ############"
$PY 12_benchmark_latency_memory.py --mode bf16 || echo "[warn] bf16 bench failed"

echo "############ 3) real INT4 (bitsandbytes — may be unavailable on aarch64) ############"
$PY 12_benchmark_latency_memory.py --mode bnb_int4 \
  || echo "[warn] bnb_int4 unavailable on this device — fall back to GGUF/TensorRT (README §4)"

echo "############ done — results in results/bench_latency_memory.jsonl ############"
cat results/bench_latency_memory.jsonl 2>/dev/null
