# VLA Quantization — Concepts & Code Guide

*Personal reference notes explaining why each technique works, followed by a map of the code in this folder.*

---

## 1. The Closed-Loop Quantization Problem

When deploying architectures like OpenVLA-OFT to power-constrained edge devices (e.g., Jetson AGX Orin, Raspberry Pi), converting 16-bit floating-point (FP16) weights to 4-bit integers (INT4/W4) is necessary to overcome memory bandwidth bottlenecks.

However, open-loop evaluations often misrepresent W4 models as "lossless." In **closed-loop** systems, the rounding errors introduced by symmetric quantization **compound** when reacting to visual perturbations.

- **FP16 baseline:** The model successfully navigates a perturbation — no impact, full task success.
- **Naive W4 failure:** The same perturbation processed through standard W4 weights can alter the continuous action trajectory enough to fail the task entirely.

This precision loss is primarily driven by **True Outliers** — massive weights (often in attention and MLP down-projection layers) that force a large quantization scale factor, effectively rounding smaller, critical micro-correction weights to zero.

> **In our data:** `layers.1.mlp.down_proj` has one channel ~10⁵× the median.
> Per-tensor INT4 sets its scale from that spike → all other values round to ~0 → 0% task success.

---

## 2. Incoherence Processing via Hadamard / DCT Transform

To prevent feature wipeout, we apply an **orthogonal transformation** before quantization.

**The mechanism:** The FP16 weight matrix `W` is multiplied by a normalized orthogonal matrix `Q` (Hadamard, DCT, SRHT, etc.):

```
W_rotated = W @ Q^T
x_rotated = Q @ x
```

So `W @ x = (W @ Q^T) @ (Q @ x)` — mathematically identical, but the representations change.

**The result:** The transform "smears" the magnitude of isolated outliers evenly across the entire vector. By reducing the maximum absolute value in any one channel, we can use a significantly **smaller scale factor** for INT4 quantization.

**The benefit:** A tighter scale factor preserves the high-resolution precision of the smaller weights during INT4 conversion. During inference, the inverse transform reconstructs the original spatial logic, preserving the fine-grained action tokens needed for successful task completion.

> **In our data:** Hadamard rotation cuts median activation outlier ratio from **18.0 → 1.6** and max outlier from **105,928 → 6.2**. INT4 per-tensor error drops from **96% → 26%**. W4A4 success goes from **0% → 88.8%**.

**Which transform to use?** We benchmarked 16. The rule is: *spread energy, don't concentrate it.*
- ✅ Works (~27% A4 error): DCT, Hadamard, SRHT, Walsh, DST, random orthogonal — global dense mixers
- ⚠️ Partial (70–90%): Butterfly, Polar, Givens — incomplete mixing
- ❌ Harmful: PCA (concentrates energy into principal axes — the *opposite* of what we need), ZCA, Haar
- **Deployment pick = DCT** — data-free, O(n log n), no power-of-two size constraint

---

## 3. Hardware Execution: Memory Bandwidth vs. Compute

Dequantizing weights back to FP16 at inference time does **not** defeat the purpose of quantization, because of the physical separation between main memory and processor registers.

**The bottleneck:** On edge devices, the primary constraint is not math latency — it is **memory bandwidth** (moving data from VRAM to the processor).

**The solution:** Weights are stored and transported as lightweight 4-bit packages, arriving at the processor **4× faster** than FP16 weights.

**On-the-fly dequantization:** Weights are unpacked using a shared FP16 block scale factor (e.g., 1 scale factor per 64 weights) only once they reach the ultra-fast processor registers. The FP16 representation exists only for a fraction of a millisecond to align with FP16 activations.

Result: you get the **storage density and transport speed** of a 4-bit model with the **numerical safety** of a 16-bit model during the actual math.

---

## 4. SIMD and the 4-bit Throughput Advantage

While a hardware clock cycle takes the same amount of time regardless of bit depth, INT4 exponentially increases processing throughput via **SIMD (Single Instruction, Multiple Data)** vectorization.

| Bit depth | Values per 64-bit register |
|---|---|
| FP16 (16-bit) | 4 values |
| INT4 (4-bit) | **16 values** |

Using specialized edge opcodes (e.g., `DP4A` on NVIDIA, `ARM NEON` on ARM), the processor's Fused Multiply-Add (FMA) unit broadcasts the shared FP16 scale factor and performs **16 simultaneous multiply-add operations** in a single clock cycle.

**Bottom line:** We achieve the storage density and transport speed of a 4-bit model, but execute the control logic with the mathematical precision of a 16-bit model.

> **Why our fake-quant numbers are honest:** Our experiments simulate quantization in FP32 (quantize → dequantize in floating point). This correctly captures the *accuracy* effect of the reduced bit depth, but does NOT realize the bandwidth/speed win. The Orin real-hardware result (86.0%, −2.2pt vs A100 bf16) confirms the accuracy transfers.

---

## 5. Component Sensitivity Summary

Not all parts of the VLA are equally sensitive:

| Component | Safe bits | Cliff | Why |
|---|---|---|---|
| Vision tower (DINOv2+SigLIP) | INT8, FP8-E4M3 | — | Visual features are robust; no outlier problem |
| Llama-2-7B backbone | W4A4 (with DCT), W3A8 | W2 | Rotation fixes activation outliers; weights are easy |
| L1RegressionActionHead | INT8 | INT4 (→70%), INT3 (→50%) | Small MLP; a few wrong action tokens = task failure |

**The floor is the action head.** Keep it at INT8 minimum.
**The backbone is the main win.** DCT-W3A8 = 74% smaller at no accuracy loss.

---

## 6. Code Map

| File | What it does |
|---|---|
| `quant.py` | Core fake-quant primitives: `FakeQuant`, `ComponentQuant`, vision/backbone/head enumerators, FP8 support |
| `quant_advanced.py` | Advanced techniques: `Transform` (DCT/Hadamard rotation grafted into backbone), AWQ, SVD-residual, GPTQ, MixedPrecision |
| `rotations.py` | All 16 orthogonal transforms: `build_transform(n, kind)` → `(M, M_inv)` |
| `capture.py` | Calibration data capture hooks (activation statistics for AWQ/GPTQ) |
| `08_quant_demo.py` | Quick interactive demo of a single quantization config |
| `08A–08E_*.py` | Layer-by-layer trial scripts and calibration capture variants |
| `_test_rot.py` | Unit tests for rotation correctness (`W @ Q^T @ Q @ x == W @ x`) |

The main experiment runners are **one level up**:

| File | What it does |
|---|---|
| `../09_quant_sweep.py` | 28+ config persistent-model sweep → leaderboard JSONL |
| `../10_outlier_ablation.py` | Per-layer outlier ratio + A4 error pre/post rotation (the mechanism paper) |
| `../11_transform_compare.py` | Fast proxy: rank all 16 transforms in minutes without rollouts |
| `../20_component_grid.py` | Component-wise grid: vision/backbone/head independence study |
