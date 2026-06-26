# Competitive Analysis — VLA Quantization (for IROS/CoRL positioning)

**Compiled:** 2026-06-26  
**Purpose:** Ground the "are we faster / better than OmegaVLA & DynVLA?" question in the actual
literature, and reposition our novelty accordingly.

---

## TL;DR — read this first

1. **"OmegaVLA" = Ω-QVLA.** **"DynVLA" = DyQ-VLA / DynamicVLA.** Both are real, recent (2026) papers.
2. **"Rotation rescues W4A4" is NO LONGER a novel finding.** Ω-QVLA (and QuaRot/SpinQuant before it)
   already established it. We **cannot** claim discovery of the mechanism. Our novelty must shift.
3. **Both competitors target *diffusion-head* VLAs (π0.5, GR00T).** We target **L1-regression OFT** —
   a *different, simpler, faster* action head with no iterative denoising. This is a genuine
   differentiator, not a weakness.
4. **They beat us on accuracy retention** (98–99.5% vs our ~92%). We must **not** compete on that axis.
5. **We must compete on: real edge hardware (Jetson Orin) + energy + transform taxonomy +
   L1-regression target.** Those are open lanes.
6. **The O(n log n) speed claim is defensible *in theory* but UNIMPLEMENTED** — our own benchmark
   runs the naive O(n²) dense path (183 ms FWHT). Claim it as an asymptotic property + future kernel,
   or build the butterfly kernel before claiming a measured speedup.

---

## Competitor 1 — Ω-QVLA ("OmegaVLA")

**Title:** Ω-QVLA: Robust Quantization for Vision-Language-Action Models via Composite Rotation and
Per-step Scaling  
**Code:** https://github.com/UCMP13753/Omega-QVLA

| Aspect | Ω-QVLA |
|---|---|
| Precision | W4A4 (uniform, no mixed precision) |
| Training | Training-free PTQ |
| Rotation | **Composite SVD–Hadamard**: SVD equalizes per-channel *weight* energy; Hadamard diffuses residual *activation* outliers |
| Activation handling | **Per-step DiT activation scaling** (absorbs dynamic-range variation across diffusion timesteps) |
| Targets | Language backbone **+ the entire diffusion action head** |
| Models | π0.5, GR00T N1.5 (both diffusion-head VLAs) |
| Results | π0.5 → **98.0%** (FP16 97.1%); GR00T N1.5 → **87.8%** (FP16 87.0%) — *matches/exceeds* FP16 |
| Memory | **−71.3%** static footprint |

**What this means for us:**
- Their "composite rotation" is the **same family** as our DCT/Hadamard rotation. We are **not** first
  to rotate-then-quantize a VLA. Reposition: we do **not** claim the mechanism; we contribute the
  **systematic 16-transform taxonomy** (they pick one: SVD-Hadamard) and the **edge deployment**.
- Their rotation has a **dense SVD component** on the weight side. Crucially, *weight-side rotations
  fold into W offline (free at inference)*. The honest open question: **what is their *activation-side*
  online rotation cost?** If it's Hadamard, it's also O(n log n) — so we **cannot** blanket-claim we're
  faster. **TODO: read their code to confirm the inference-time activation rotation cost before any
  "faster" claim.**
- They target **diffusion heads**; their whole "per-step DiT scaling" is meaningless for our
  L1-regression head (no timesteps). So our method and theirs **apply to different architectures** —
  a clean way to avoid a head-to-head we'd lose on accuracy.

## Competitor 2 — DyQ-VLA ("DynVLA" / "DynamicVLA")

**Title:** DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models  
**arXiv:** 2603.07904 (March 2026)

| Aspect | DyQ-VLA |
|---|---|
| Approach | **Dynamic / per-timestep** quantization — sensitivity-aware bit-width *switching* |
| Trigger | Real-time **kinematic proxies** decide when to switch bit-width |
| Allocation | Kinematic-guided module picks optimal bit-width per stage |
| Memory | **30.9%** of original footprint |
| Performance | **99.5%** of FP16 retained |
| Speedup | **1.49× simulation, 1.43× real-world** — *measured*, not fake-quant |

**What this means for us:**
- **They have a *measured* real-world speedup (1.43×). We do not** (we run fake-quant). This is our
  biggest credibility gap. A reviewer will compare 1.43× measured vs. our "theoretical O(n log n)."
  **We must get a real packed-kernel speedup number on the Orin** (GGUF/TensorRT-LLM) or we lose this.
- Their novelty is **runtime dynamic bit allocation** — orthogonal to ours (static rotation+quant).
  We are not competing with their mechanism; we could even cite them as complementary.

---

## Where we actually stand (honest scorecard)

| Axis | Ω-QVLA | DyQ-VLA | **Us (current)** | Verdict |
|---|---|---|---|---|
| Accuracy retention | 98–101% | 99.5% | **~92%** (88.8/96) | ❌ behind — don't compete here |
| Measured speedup | n/r | **1.43× real** | none (fake-quant) | ❌ gap — must fix |
| Memory reduction | 71.3% | 69.1% | 62% (2.63×) | ➖ comparable |
| Real edge hardware | sim | partial | **Jetson Orin, real** | ✅ our lane |
| Energy (J/inference) | no | no | **(measuring)** | ✅ open lane |
| Transform taxonomy | 1 (SVD-Had) | n/a | **16 transforms** | ✅ our contribution |
| Action-head family | diffusion | diffusion | **L1-regression OFT** | ✅ differentiator |

## Repositioned novelty statement (use this in the paper)

> Prior VLA quantization (Ω-QVLA, DyQ-VLA) establishes that rotation- and dynamic-bit schemes can push
> *diffusion-head* VLAs (π0.5, GR00T) to W4A4 in **simulation**. We instead study **L1-regression
> OpenVLA-OFT** — a non-diffusion action head — and ask which of **16 structured transforms** best
> conditions its backbone for W4A4, then **deploy the result on a $2k Jetson AGX Orin** with **real
> latency, control-frequency, and energy** measurements. Our contribution is not the rotation mechanism
> (known) but (i) a transform taxonomy that tells practitioners *which* rotation to use and why, and
> (ii) the first **measured edge-deployment** characterization (24 Hz effective control via 8-action
> chunks, J/inference) of a quantized L1-regression VLA.

## Concrete to-dos this analysis creates

1. **DROP** any "we discovered rotation fixes W4A4" framing — cite Ω-QVLA/QuaRot/SpinQuant for it.
2. **READ Ω-QVLA code** (github.com/UCMP13753/Omega-QVLA) to confirm their *inference-time* activation
   rotation cost before any "we're faster" sentence.
3. **GET A REAL SPEEDUP NUMBER** on the Orin (packed INT4 kernel) — DyQ-VLA's measured 1.43× is the bar.
4. **MEASURE ENERGY** (tegrastats J/inference) — nobody else reports it; uncontestable edge lane.
5. **LEAD with L1-regression + edge + taxonomy**, not accuracy retention (we lose that axis).
6. **Frame O(n log n) honestly**: asymptotic property of pure structured transforms vs. composite
   SVD-rotation; note the fast kernel is implemented/future-work as appropriate.

---

## Sources
- [Ω-QVLA: Robust Quantization for VLA via Composite Rotation and Per-step Scaling](https://deeplearn.org/arxiv/761767/%CF%89-qvla:-robust-quantization-for-vision-language-action-models-via-composite-rotation-and-per-step-scaling) · [code](https://github.com/UCMP13753/Omega-QVLA)
- [DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied VLA (arXiv 2603.07904)](https://arxiv.org/abs/2603.07904)
- [DynamicVLA (arXiv 2601.22153)](https://arxiv.org/abs/2601.22153)
- [QuantVLA: Scale-Calibrated PTQ for VLA (arXiv 2602.20309)](https://arxiv.org/html/2602.20309)
- [EaqVLA: Encoding-aligned Quantization for VLA (arXiv 2505.21567)](https://arxiv.org/html/2505.21567v1)
- [HBVLA: 1-Bit PTQ for VLA (arXiv 2602.13710)](https://arxiv.org/pdf/2602.13710)
- [SpinQuant: LLM Quantization with Learned Rotations](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf)
