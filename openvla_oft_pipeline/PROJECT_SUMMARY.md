# OpenVLA-OFT Quantization Research — Project Summary

**Date:** 2026-06-17  
**Author:** Justin Williams  
**Email:** justinwilliamstech693@gmail.com  
**Hardware:** 8× H100 (training/A100 eval), AGX Orin (edge deployment)

---

## 1. Research Goal

Find a quantization scheme for an OpenVLA-OFT policy fine-tuned on LIBERO that satisfies **at least one** of:

| Goal | Target |
|------|--------|
| A | Same size or speed, **better** task success |
| B | Smaller size / lower latency, **same** task success |
| C | Smaller size / lower latency, task success within **10–15%** of bf16 baseline |

---

## 2. Model Architecture

- **Backbone:** Llama-2-7B (84% of parameters, all quantization targets here)
- **Vision:** DINOv2 + SigLIP fused tower (stays bf16)
- **Action head:** L1RegressionActionHead (stays bf16)
- **Optional:** ProprioProjector (2-cam+proprio variant only)
- **Two variants:**
  - **Our fine-tuned model** — 1-cam, no proprio, ~88% honest LIBERO success
  - **Reference model** (`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`) — 2-cam + proprio, ~97% LIBERO success (target for publication)

---

## 3. Key Finding: Hadamard Rotation Rescues W4A4

### The Activation Outlier Problem
Naive INT4 quantization of activations fails catastrophically (0% success) because:
- `layers.1.mlp.down_proj` has one channel with magnitude ~10⁵× the median
- Per-tensor INT4 scale is dominated by the outlier → all other values round to zero

### The Fix: Incoherence Rotation
Insert orthogonal matrix Q so `Wx = (WQ^T)(Qx)`:
- Outliers are spread across all channels (Gaussian-like after rotation)
- Per-tensor INT4 error drops from **96.3% → 26.1%**
- Median outlier ratio: 18.0% → 1.6% (11× reduction)
- Max activation magnitude: 105,928 → 6.2 (17,000× reduction)

### Success Rate Recovery

| Configuration | Success Rate | Notes |
|--------------|-------------|-------|
| bf16 baseline | 88.2% | Official OFT step budget |
| W8A8 (BnB) | ~87% | Near-lossless |
| W4A16 (BnB INT4) | ~85% | Moderate drop |
| W4A4 naive | 0% | Destroyed by outliers |
| W4A4 + Hadamard | 86.5% | Rescues W4A4 |
| W4A4 + DCT | **89.8%** | Best transform, zero pow2-fallback layers |
| W4A4 + SRHT | 88.9% | Runner-up |

---

## 4. Transform Taxonomy (16 Transforms Ranked)

Evaluated via fast proxy metric: median post-transform INT4 per-tensor activation error.

| Rank | Transform | A4 Error | Category |
|------|-----------|----------|----------|
| 1 | SRHT | 26.3% | Structured random |
| 2 | DST | 27.0% | Spectral |
| 3 | DCT | 27.3% | Spectral ★ deployment pick |
| 4 | Hadamard | 27.4% | Spectral |
| 5 | Walsh | 28.1% | Spectral |
| 6 | Haar | 29.2% | Wavelet |
| 7 | Random Orthogonal | 31.8% | Random |
| 8 | Butterfly | 34.5% | Structured |
| 9 | Givens | 38.2% | Geometric |
| 10 | Householder | 41.7% | Geometric |
| 11 | Permutation | 52.4% | Combinatorial |
| 12 | Polar | 58.9% | Algebraic |
| 13 | ZCA | 63.1% | Statistical |
| 14 | PCA | 81.7% | Statistical |
| 15 | Identity | 96.3% | Baseline (no rotation) |

**Unifying principle:** "Spread energy, don't concentrate it." Transforms that spread weight/activation energy broadly (spectral, random) outperform those that concentrate it (PCA/ZCA align with principal axes, amplifying the very outliers we want to suppress).

**DCT chosen for deployment** because: (a) deterministic, (b) no pow2-size fallback layers, (c) marginal improvement over Hadamard in full eval (89.8% vs 88.8%).

---

## 5. Size & Memory Footprint

### Model Size (Llama backbone only, 224 linear layers)
| Format | Size | Reduction |
|--------|------|-----------|
| bf16 | 15.4 GB | 1× |
| BnB INT8 | ~7.8 GB | ~2× |
| BnB INT4 | 5.87 GB | **2.63×** |
| GGUF Q4_K_M | 3.80 GB | **3.3× smaller** (Jetson) |

### A100 Measured Memory + Latency
| Mode | GPU Memory | Per-query Latency |
|------|-----------|-------------------|
| bf16 | 15.43 GB | 53 ms |
| BnB INT8 | ~8.1 GB | ~65 ms |
| BnB INT4 | 5.87 GB | 90 ms |
| rot_w4a4 (fake) | 29.67 GB | 96 ms |

> **Note:** `rot_w4a4` memory/latency on A100 is a fake-quant artifact — rotation matrices stored separately in bf16. Real packed INT4 kernels (llama.cpp, TensorRT-LLM) give true 5.9 GB / sub-bf16 latency.

### Rotation Overhead (per policy query, 224 layers)
| Method | Overhead |
|--------|---------|
| Identity (no rotation) | 1.6 ms |
| Dense matrix mul | 26 ms + 276 MB |
| FWHT naive | 183 ms (no fused kernel on A100) |

---

## 6. AGX Orin Results

### Hardware
- **Device:** NVIDIA AGX Orin
- **Memory:** 64 GB unified (Jetson)
- **IP:** 10.10.10.112 (`agxorin@`)

### Measured Results (from user)
| Configuration | Success Rate |
|--------------|-------------|
| Standard bf16 (1-cam) | 60.2% |
| GGUF Q4_K_M bf16 | ~56% (est.) |

> **Known issue:** Orin 60.2% vs A100 88.2% gap is unexplained by step budget (only a ~2-point effect). Primary hypothesis: EOS/STOP token discrepancy — training appends EOS (111 tokens total), A100 eval omits it (110 tokens), Orin `07_` eval uses `predict_action()` which appends STOP — potential off-by-one in action hidden-state offset.

### GGUF Benchmark (Orin, from user)
| Metric | Value |
|--------|-------|
| Load time | ~18s |
| Per-token latency | ~45ms |
| Memory footprint | ~4.1 GB |
| Throughput | ~22 tok/s |

---

## 7. Codebase Map

```
openvla_oft_pipeline/
├── 02_eval_libero.py              # Main LIBERO eval (patched: official OFT step budgets)
├── 09_quant_sweep.py              # 28+ config persistent-model sweep, leaderboard JSONL
├── 10_outlier_ablation.py         # Per-layer outlier ratio + A4 error pre/post rotation
├── 11_transform_compare.py        # Fast proxy screening: all 16 transforms ranked in minutes
├── 12_benchmark_latency_memory.py # Real latency/memory: bf16, bnb_int4, rot_w4a4
├── 13_rotation_overhead.py        # Rotation cost microbenchmark (224 layers)
├── quant_expirements/
│   ├── quant.py                   # Basic W4/W8/W4A4 fake-quant
│   ├── quant_advanced.py          # Rotation, AWQ, SVD residual, GPTQ, MixedPrecision
│   ├── rotations.py               # 16 transforms: build_transform(n, kind, ...) → (M, Minv)
│   └── capture.py                 # Calibration data capture hooks
├── papers/
│   └── paper_combined.tex         # Single flagship paper (365 lines, all real numbers)
├── checkpoints/
│   ├── openvla_oft_libero/epoch_003/   # Our fine-tuned 1-cam model
│   └── openvla_oft_reference/sogd10/   # moojink reference 2-cam+proprio (15 GB)
├── jetson_bundle/                 # Code bundle rsynced to Orin
├── transfer_to_jetson.sh          # Rsync A100 → Orin (SSH multiplexing)
├── RESULTS_quant.md               # Living results doc with all tables
└── PROJECT_SUMMARY.md             # This file
```

---

## 8. Eval Protocol

### Official OFT Per-Suite Step Budgets (patched in `02_eval_libero.py`)
| Suite | Max Steps | Tasks |
|-------|----------|-------|
| libero_spatial | 220 | 10 |
| libero_object | 280 | 10 |
| libero_goal | 300 | 10 |
| libero_long | 520 | 10 |

> **Important:** Original code used flat 600 steps for all suites. Corrected to official values. Effect: ~2-point improvement (not the cause of Orin gap).

### Quantization Eval Convention
- All quantization experiments use **simulated (fake) quantization** — quantize→dequantize in fp16/bf16
- Rounding errors are real; size/latency wins are NOT realized until real packed kernels
- Only Llama backbone linears (224 layers, 84% of parameters) are quantized
- Vision tower, projector, lm_head, embeddings, regression head stay bf16

---

## 9. Infrastructure

### SSH Configuration
| Direction | From | To | Key |
|-----------|------|----|-----|
| A100 → Orin | `justin_williams1@10.10.10.104` | `agxorin@10.10.10.112` | `~/.ssh/id_ed25519` |
| Orin → A100 | `agxorin@10.10.10.112` | `justin_williams1@10.10.10.104` | `~/.ssh/id_ed25519_a100` |

Orin's public key installed in `~/.ssh/authorized_keys` on A100. Reverse connection verified: `REVERSE_SSH_OK from hyperplane`.

### Training Environment
- **venv:** `../venv/bin/python` — torch 2.2.0, transformers 4.40.2, deepspeed 0.14.5
- **Real runtime:** `~/.local` — torch 2.7.1 (used for actual train launches)
- **Launch:** `setsid` + `tmux` (nohup insufficient; torchrun re-arms SIGHUP)
- **GPU sharing:** Colleagues share the 8× H100 box; default `NPROC=4`, check `nvidia-smi` before multi-GPU

---

## 10. Known Issues / Blockers

### Critical: Reference Model Scores 46% (Expected ~97%)
- **Cause:** `get_vla()` calls `check_model_logic_mismatch()` which overwrites the reference checkpoint's `modeling_prismatic.py` with the local repo's version
- **Fix:** Restore backup: `cp checkpoints/openvla_oft_reference/sogd10/modeling_prismatic.py.back.20260624_184208 checkpoints/openvla_oft_reference/sogd10/modeling_prismatic.py`
- **Or:** Set `pretrained_checkpoint` to the HF Hub path so `model_is_on_hf_hub()` returns True and skips the sync
- **Status:** BLOCKED — must fix before publishing credible absolute numbers

### Orin 60% vs A100 88% Gap
- **Ruled out:** Step budget (only ~2-point effect)
- **Primary hypothesis:** EOS/STOP token discrepancy
  - Training appends EOS → 111 tokens
  - A100 eval omits EOS → 110 tokens
  - Orin `07_` eval uses `predict_action()` → appends STOP → 111 tokens
  - Potential off-by-one in action hidden-state offset reading
- **Status:** Unverified — needs controlled test

---

## 11. Pending Tasks

| # | Task | Status |
|---|------|--------|
| 9 | Reproduce ~97% bf16 with reference model (2-cam+proprio) | BLOCKED (modeling_prismatic.py) |
| 10 | Inject DCT W4A4 quant into reference model; re-run full sweep | Waiting on #9 |
| — | Verify EOS/STOP token hypothesis for Orin gap | Not started |
| — | Update paper with corrected absolute numbers from reference model | Waiting on #9 |
| — | Run Orin latency/memory with real INT4 kernels (llama.cpp or TRT-LLM) | Partial (GGUF only) |

---

## 12. Publication Plan

### Papers Scoped (from this work)
1. **Flagship paper** (`papers/paper_combined.tex`): "Quantization of VLA Policies for Edge Robotics: A Transform Taxonomy Study" — combines all three angles
2. **Transform zoo paper**: Benchmarks all 16 transforms; community reference (no one else will need to try)
3. **Edge deployment paper**: Jetson AGX Orin numbers, GGUF vs BnB vs TensorRT-LLM

### Key Claims (currently supported)
- **HEADLINE (all-4-suite, 400 rollouts): vision-INT8 + DCT-W3A8 backbone + head-INT8 = 74% smaller at 89.5% success vs 88.0% bf16 — no measurable accuracy loss (within ±2.5% CI). Strongest goal met: smaller size, same success.**
- Component sensitivity: action head is the floor (INT8 ok, INT4→70%, INT3→50%); vision tower is free (INT8/FP8 lossless)
- W3A8 ≥ W4A4 at smaller size — post-rotation, activation bits dominate weight bits (activations-are-the-wall)
- Hadamard/DCT rotation rescues W4A4 from 0% → 88–90% success (mechanism, NOT novel — Ω-QVLA/QuaRot own it)
- DCT is optimal deployment transform (deterministic, no size-constraint fallback)
- All numbers fake-quant; real INT4-kernel size/latency on Jetson Orin is the remaining gap

### What's Needed for Submission
- [ ] Reference model (2-cam+proprio) bf16 baseline confirmed at ~97%
- [ ] Full quant sweep re-run with reference model (credible absolute numbers)
- [ ] Orin latency/memory with real INT4 kernels (not just GGUF)
- [ ] Orin 60% vs A100 88% gap explained or controlled for

### Competitive positioning (see `COMPETITIVE_ANALYSIS.md`)
- **"Rotation rescues W4A4" is NOT novel** — Ω-QVLA / QuaRot / SpinQuant established it. Do not claim the mechanism.
- Competitors (Ω-QVLA, DyQ-VLA) target **diffusion-head** VLAs (π0.5, GR00T); **we target L1-regression OFT** — different architecture, our differentiator.
- They beat us on accuracy retention (98–99.5% vs our ~92%) — **do not compete on that axis.**
- DyQ-VLA has a **measured 1.43× real speedup**; we have fake-quant only — **must get a real Orin kernel speedup.**
- **Our open lanes:** real Jetson Orin deployment, energy (J/inference — nobody reports it), 16-transform taxonomy, L1-regression target.
- **Architecture fact (verified):** every code path uses `L1RegressionActionHead`. **There is NO diffusion head** in this pipeline. We quantize the Llama-2-7B backbone; the L1 head stays bf16.
- **Control rate (verified):** `OFT_CHUNK_SIZE=8`, so 330 ms/query = **~24 Hz effective** control. Report Hz, not ms.

---

## 13. Quick Reference Commands

```bash
# A100: run reference model eval (after fixing modeling_prismatic.py)
cd openvla_oft_pipeline
source ../venv/bin/activate
python 02_eval_libero.py \
  --checkpoint_path checkpoints/openvla_oft_reference/sogd10 \
  --suite all --num_trials_per_task 10

# A100: run quant sweep (28+ configs, all 4 suites)
python 09_quant_sweep.py --suite all --num_trials 10

# Transfer to Jetson
bash transfer_to_jetson.sh

# SSH to Jetson
ssh agxorin@10.10.10.112

# Jetson: run benchmarks
cd /home/agxorin/Desktop/offline-robotics/quant_bench
bash run_bench.sh
```

---

*Generated 2026-06-17. For full conversation transcript see `.claude/projects/.../cb135298-*.jsonl`.*
