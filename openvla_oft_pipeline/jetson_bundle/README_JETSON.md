# Jetson AGX Orin — latency & memory benchmark for quantized OpenVLA-OFT

Goal: measure **real** per-policy-query latency and memory footprint of the W4A4 / bf16
policy on the edge target (`agxorin@10.10.10.112`), to turn the simulated A100 numbers into
deployment numbers. The benchmarks use **synthetic inputs** — no LIBERO/MuJoCo/datasets
needed.

## 0. Expected layout on the Jetson
```
/home/agxorin/Desktop/offline-robotics/quant_bench/
  12_benchmark_latency_memory.py      # real latency + memory per mode
  13_rotation_overhead.py             # rotation cost (pure torch, no model)
  02_eval_libero.py  02A_eval_quant_libero.py   # load_model / load_quantized_model
  quant_expirements/{quant,quant_advanced,rotations,capture}.py
  openvla-oft/prismatic/...           # the model class source (on sys.path)
  checkpoints/openvla_oft_libero/epoch_003/{hf_model/, action_head.pt, action_stats.json}
  hf_cache/transformers_modules/openvla/...      # HF remote-code (place per step 1)
```
The transfer script (`transfer_to_jetson.sh`, run on the A100 box) puts these in place.

## 1. One-time placement of the HF remote-code cache
The checkpoint's `config.json` references HF remote code
(`openvla/openvla-7b--modeling_prismatic`). With `HF_HUB_OFFLINE=1` that must exist locally:
```bash
mkdir -p ~/.cache/huggingface/modules/transformers_modules
cp -r /home/agxorin/Desktop/offline-robotics/quant_bench/hf_cache/transformers_modules/openvla \
      ~/.cache/huggingface/modules/transformers_modules/
```

## 2. Python environment (do NOT copy the x86 venv)
Use the JetPack/NVIDIA aarch64 PyTorch build (matching the device's CUDA/JetPack), then:
```bash
python3 -m venv ~/venv_quant && source ~/venv_quant/bin/activate
# 1) install the NVIDIA aarch64 torch wheel for your JetPack (see NVIDIA Jetson PyTorch index)
# 2) pinned deps that match the checkpoint:
pip install "transformers==4.40.2" "tokenizers==0.19.1" "accelerate>=0.30" \
            "safetensors" "numpy<2" "pillow" "timm" "sentencepiece" "protobuf"
```
Verify: `python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`

## 3. Run the benchmarks
```bash
cd /home/agxorin/Desktop/offline-robotics/quant_bench
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash run_bench.sh           # runs rotation-overhead + bf16 + (attempts) bnb_int4
```
Results append to `results/bench_latency_memory.jsonl`. Compare against the A100 baseline
in the paper.

## 4. INT4 on Jetson — the real-kernel caveat
`bitsandbytes` aarch64 support is fragile; `--mode bnb_int4/bnb_int8` **may fail to build/import
on the Jetson.** Order of attack:
1. **bf16 + rotation overhead first** (always works) → real edge bf16 latency/memory + the
   rotation cost on memory-bound hardware (where structured O(n log n) should finally beat dense).
2. **Try bnb INT4** — if it imports, that's a real packed-4-bit memory + latency number.
3. **If bnb fails**, use an edge-native INT4 path for the deployment number:
   - **llama.cpp / GGUF** (you already have GGUF tooling) — strong on Jetson, real INT4 kernels.
   - **TensorRT-LLM** — best latency on Jetson but more integration work.
   - **AWQ** (`autoawq`) kernels — aarch64 wheels are hit-or-miss.

The memory footprint (weight packing) is what transfers cleanly; the **latency win is the
memory-bandwidth win on this device**, which is exactly what we want to demonstrate here.

## 5. Memory headroom
bf16 weights = 15.4 GB. AGX Orin has 32 GB or 64 GB unified memory — bf16 fits on 64 GB,
is tight on 32 GB (close other processes). INT4 (~5.9 GB) fits comfortably on either.
Avoid `--mode rot_w4a4` on a 32 GB device: the current fake-quant impl stores bf16 weights
plus per-layer rotation matrices (~30 GB) and will OOM; it is for accuracy on the A100, not
for the edge footprint (use bnb/GGUF INT4 for the real edge number).
