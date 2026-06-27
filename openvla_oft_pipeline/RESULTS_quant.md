# Low-bit quantization of OpenVLA-OFT on LIBERO — results

**Model:** `checkpoints/openvla_oft_libero/epoch_003` (Llama-2-7B backbone + L1RegressionActionHead).
**Baseline:** bf16, **90.25%** average closed-loop success over the 4 LIBERO suites (400 rollouts).
**Method:** simulated (fake) quantization of the backbone linears only (vision tower / projector /
lm_head / action head stay bf16); scored by closed-loop LIBERO success.
**Code:** `quant_expirements/quant_advanced.py`, `09_quant_sweep.py`, `10_outlier_ablation.py`.

## 1. Headline

> **Hadamard rotation enables W4A4 (4-bit weights *and* activations) at 88.8% — 98.4% of the
> bf16 baseline — where naive per-tensor W4A4 collapses to 0%. The quantization bottleneck for
> VLA policies is activation outliers, not weights.**

## 2. Full-eval confirmation (400 rollouts each)

| config | success | retention | eff. bits | note |
|---|---|---|---|---|
| bf16 baseline | 90.25% | — | 16 | reference |
| `uniform_w4` (naive) | 90.75% | 100.6% | 4.0 | weights are easy; naive W4 ≈ lossless |
| `rot_w4_had` | 89.0% | 98.6% | 4.0 | rotation adds nothing over naive at W4-weights |
| **`rot_w4a4_had`** | **88.8%** | **98.4%** | 4.0 | **W4A4 — naive = 0%. The result.** |
| `rot_w3a8_had` | 88.8% | 98.4% | 3.0 | 3-bit weights, 8-bit activations |
| `uniform_w4a4` (naive) | **0%** | 0% | 4.0 | activation outliers destroy per-tensor A4 |

The three rotation finalists are statistically indistinguishable (±2.5% CI at 400 rollouts).

### Per-suite breakdown (success %, vs bf16 baseline)

| suite | bf16 | rot_w4a4_had | rot_w4_had | rot_w3a8_had |
|---|---|---|---|---|
| libero_spatial | 87 | 87 | 88 | 87 |
| libero_object | 99 | 97 | 99 | 98 |
| libero_goal | 86 | 88 | 86 | 84 |
| **libero_long** | **89** | **83** | **83** | **86** |
| **average** | **90.25** | **88.8** | **89.0** | **88.8** |

The quantization cost is concentrated almost entirely in **libero_long** (−3 to −6 pts) —
long-horizon rollouts accumulate per-step quantization error; short-horizon suites are
preserved within noise. Targeted recovery effort should focus on the long-horizon regime.

## 3. Screening sweep (libero_spatial, 6 tasks × 5 rollouts) — what each method showed

- **Weights down to W3 are easy** for every method (uniform / rotation / AWQ / GPTQ all ~80%).
- **Pure W2 is a wall** — `uniform`, `rotation`, `AWQ`, `SVD-residual` (rank 64 & 128) all → **0%**.
  Only untested escape is a fundamentally higher-capacity scheme.
- **Activations (A4) are the wall** — naive A4 → 0%; **rotation rescues it**.
- **GPTQ / AWQ match but never beat rotation**; **GPTQ + rotation stacked slightly *hurt*** (basis interaction).
- **SVD low-rank residual** doesn't help: redundant at W3, insufficient-rank at W2.
- **Mixed precision** works but is dominated (more bits, same success).

## 4. Mechanism (`10_outlier_ablation.py`, 32 calib frames, 224 backbone linears)

| metric | pre-rotation | post-rotation | factor |
|---|---|---|---|
| median activation outlier ratio (max\|·\|/median\|·\|) | 18.0 | 1.6 | 11× |
| **max** activation outlier ratio | 105,928 | 6.2 | 17,000× |
| median A4 per-tensor quant rel-error | 96.5% | 26.1% | 3.7× |

Worst offender: `language_model.model.layers.1.mlp.down_proj` — one channel ~10⁵× the median
(a "massive activation"). Per-tensor A4 quant sets its single scale from that spike, leaving the
bulk of channels ~0 levels (→ 0% success). Rotation mixes it into a near-Gaussian spread (→ 88.8%).

## 4b. Transform zoo — which rotation conditions activations best (mechanism proxy)

`11_transform_compare.py` ranks 16 transforms by median A4 per-tensor activation error after
the transform (lower = better; identity = 96.3%). Fast offline screen over 32 calib frames.

| tier | transforms (A4 err %) | category |
|---|---|---|
| ✅ works (~27%) | srht 26.3, dst 27.0, dct 27.3, hadamard/walsh 27.4, hartley 27.5, random-orth 28.2 | global dense orthogonal mixing |
| ⚠️ partial (70–90%) | polar 70.7, butterfly 71.0, givens 89.9 | incomplete mixing |
| ❌ fail/harmful (90–97%) | haar 90.9, pca 81.7 (outlier→171k), identity 96.3, permutation 96.3, householder 96.4, zca 96.6 | localized / permuting / energy-compacting |

### Transform champions — FULL EVAL (400 rollouts) @ W4A4, vs bf16 90.25%

| transform | full eval | retention | pow2-fallback layers |
|---|---|---|---|
| random_orthogonal | 90.5% | 100.3% | 0 |
| dct | 89.8% | 99.5% | 0 (true transform on all 224) |
| srht | 89.8% | 99.5% | 32 |
| hadamard (rot_w4a4_had) | 88.8% | 98.4% | 32 |

All global mixers are statistically tied at ~89–90% (±2.5% CI) — confirms the proxy ranking
at full resolution and that "any full orthogonal mixer works" at the 4-bit floor. **DCT is the
deployment pick**: full-eval-confirmed at 89.8% with ZERO fallback layers (data-free, fast,
works on any dimension incl. the 11008-dim down_proj).

**Unifying principle:** low-bit activation quant needs energy *spread uniformly* across channels.
Global orthogonal mixing delivers it; permutation (no mixing), Haar (localized), few Householder/
Givens (partial), PCA (concentrates energy — opposite), and ZCA (non-orthogonal) all violate it.
**Deployment pick = DCT** — ties Hadamard, works on any dimension (no pow2 fallback), fast to build.

## 4c. Component-wise grid (`20_component_grid.py`) — vision / backbone / head

### ★ HEADLINE — paper-grade ALL-4-SUITE eval (10 tasks × 10 rollouts × 4 suites = 400 rollouts)

| config | all-suite success | retention vs bf16 88.0% | size | reduction |
|---|---|---|---|---|
| bf16 baseline | 88.0% | 100% | 15.39 GB | — |
| **`combo_v8_b3a8dct_h8`** (vision INT8 + DCT-W3A8 + head INT8) | **89.5%** | **101.7%** | **3.98 GB** | **74%↓** |
| `combo_v8_b4dct_h8` (vision INT8 + DCT-W4A4 + head INT8) | 86.2% | 98.0% | 4.79 GB | 69%↓ |

**The deployment pick `combo_v8_b3a8dct_h8` is 74% smaller at NO measurable accuracy loss**
(89.5% vs 88.0% bf16, within the ±2.5% binomial CI at 400 rollouts — i.e. statistically
indistinguishable from bf16, *not* "better"). This satisfies the strongest project goal:
**smaller size at same success.** The W4A4 variant gives a second frontier point: 69% smaller at
98.0% retention. (Fake-quant; real INT4-kernel size/latency on Jetson is the remaining gap.)

---

Treats OFT as three independently-quantizable components. Mono-cam epoch_003, fake-quant.
**Component sensitivity** (libero_spatial, 4 tasks × 5 rollouts screen — isolate one, others bf16):

| component | precision | success | takeaway |
|---|---|---|---|
| action head | INT8 | 100% | safe |
| action head | INT4 | **70%** | cliff — head is the fragile component |
| action head | INT3 | 50% | catastrophic |
| vision tower | INT8 / FP8-E4M3 | 100% / 100% | free |
| vision tower | FP8-E5M2 | 95% | slightly worse (fewer mantissa bits) |
| backbone | DCT-W4A4 | 100% | the main win |

**Pareto frontier — FULL EVAL (libero_spatial, 10 tasks × 10 rollouts), real retention vs bf16 = 86.0%:**

| config (vision + backbone + head) | success | size | reduction | retention |
|---|---|---|---|---|
| bf16 baseline | 86.0% | 15.39 GB | — | 100% |
| INT8 + INT8 + INT8 (`combo_full_int8`) | 85.0% | 8.03 GB | 48%↓ | **98.8%** |
| bf16 + DCT-W4A4 + bf16 (`backbone_w4a4_dct`) | 84.0% | 5.67 GB | 63%↓ | **97.7%** |
| INT8 + DCT-W4A4 + INT8 (`combo_v8_b4dct_h8`) | 82.0% | 4.79 GB | **69%↓** | **95.3%** |
| FP8 + DCT-W4A4 + INT8 (`combo_vfp8_b4dct_h8`) | 82.0% | 4.79 GB | 69%↓ | 95.3% |

**Deployment pick = `combo_v8_b4dct_h8`** (vision INT8 + backbone DCT-W4A4 + head INT8):
**69% smaller at 95.3% retention** on full spatial.

**Phase 1c — pushing past 70% (full eval, spatial 10×10):**

| config | success | size | reduction | retention | clears >70%↓ & >95%? |
|---|---|---|---|---|---|
| `combo_v8_b3a8dct_h8` (vision INT8 + **DCT-W3A8** + head INT8) | 84.0% | 3.98 GB | **74%↓** | **97.7%** | ✅ **YES** |
| `combo_v8_b3a4dct_h8` (vision INT8 + DCT-W3**A4** + head INT8) | 83.0% | 3.98 GB | 74%↓ | 96.5% | ✅ YES |
| `combo_vfp8_b3a8dct_h8` (vision **FP8** + DCT-W3A8 + head INT8) | 83.0% | 3.98 GB | 74%↓ | 96.5% | ✅ YES |

**The entire W3-backbone family clears the threshold: 74% smaller at 96.5–97.7% retention.** It
*beats* the W4A4 backbone on size, and A8 vs A4 (84% vs 83%) is **within the ±8% CI** at 100
rollouts — because the DCT rotation already conditions the activations, so once rotated, 4-bit vs
8-bit activations barely differ. The thesis holds twice over: low-bit activations are only the wall
*without* rotation (naive A4 = 0%); *with* DCT, even W3A4 holds 96.5%, so weights can go to 3 bits
essentially for free. **Robust frontier, not a single lucky point.** *(In progress: paper-grade
all-4-suite 10×10 for `b3a8` and `b4dct`.)*

## 5. Mapping to the goal (one of: same size/speed+better; smaller+same; smaller+10–15% worse)

- `rot_w4a4_had` and `rot_w3a8_had` → **smaller size *and* speed, same success** (within ~1.5 pts).
  W4A4 = 4× on both axes on real INT4 hardware; W3 weights ≈ 5.3×.

## 6. Status & next steps (for publication)

**Done:** method matrix, full-eval confirmation, mechanism ablation. → workshop-publishable core.
**For a main venue (CoRL/ICRA):**
1. **Realize one config (W4A4) with a real INT4 kernel on Jetson Orin Nano** — measured latency +
   memory. (The size/speed claim is currently *simulated*; this is the biggest gap.)
2. Second model or benchmark; 2–3 seeds for error bars.
3. Head-to-head vs QVLA (2602.03782) / QuantVLA (2602.20309).
