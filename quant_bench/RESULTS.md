# OFT Quantization / Edge-Deployment — Living Results Table

## ⚡ RESOLVED (2026-06-25): the 60-vs-88 gap was the transformers package, not the edge
Running the *identical* `02_eval` on the Orin reproduced **40% (fork) → 93.3% (stock transformers 4.40.2)**, matching the A100's 88%. **Root cause: the Orin had the moojink bidirectional-attention fork installed (for `07_`), which silently corrupts `02_eval`'s causal last-56 hidden-state read.** Ruled out along the way (all gave ~30-67%, none fixed it): step count, center-crop, dtype (bf16/fp16/fp32), mujoco version (2.3.7 vs 3.9.0). **The Orin is FULLY FAITHFUL — no edge/hardware degradation.** Correct stack = stock transformers 4.40.2 + mujoco 3.9.0 + `02_eval` (causal, no-STOP, last-56). All "60.2% bf16" numbers below are SUPERSEDED (fork-corrupted); real bf16 baseline ≈ 88-93%.



Keep ALL numbers here so protocols don't get conflated. Two distinct models:
- **ours (weak, 1-cam, no-proprio)** = `checkpoints/openvla_oft_libero/epoch_003/hf_model` (norm_stats has NO `libero` key → needs the bespoke `07_` eval, not standard `run_libero_eval.py`).
- **reference (2-cam + proprio)** = the standard/stronger OFT checkpoint (not on this Jetson).

## ✅ Corrected accuracy — n=100 final (matched stack: stock tf 4.40.2 + mujoco 3.9.0, `02_eval`, 220 steps)

| Config | libero_spatial | n | Notes |
|---|---|---|---|
| A100 reference | ~88% | — | standard OFT eval, center-crop |
| **Orin bf16** | **83.0% (83/100)** | 100 | 10 tasks × 10 rollouts |
| **Orin W4 plain (RTN)** | **89.0% (89/100)** | 100 | +6pt vs bf16 (within noise) |
| **Orin W4 + DCT rotation** | **86.0% (86/100)** | 100 | +3pt vs bf16 (within noise) |

Per-task breakdown (10 rollouts each):

| Task | Description | bf16 | W4 plain | W4+DCT |
|---|---|---|---|---|
| 0 | bowl between plate and ramekin | 10/10 | 10/10 | 10/10 |
| 1 | bowl next to ramekin | 8/10 | 10/10 | 10/10 |
| 2 | bowl from table center | 9/10 | 10/10 | 10/10 |
| 3 | bowl on cookie box | 9/10 | 10/10 | 10/10 |
| 4 | bowl in top drawer of cabinet | 9/10 | 10/10 | 8/10 |
| 5 | bowl on ramekin | **0/10** | **0/10** | **0/10** |
| 6 | bowl next to cookie box | 10/10 | 10/10 | 10/10 |
| 7 | bowl on stove | 10/10 | 10/10 | 10/10 |
| 8 | bowl next to plate | 8/10 | 9/10 | 8/10 |
| 9 | bowl on wooden cabinet | 10/10 | 10/10 | 10/10 |

**Key findings:**
- Task 5 ("bowl on ramekin") is 0/10 for all configs — a hard scene for this checkpoint regardless of precision.
- W4 plain (+6pt) and W4+DCT (+3pt) both exceed bf16, but differences are within statistical noise at n=100 (±~4pt at 95% CI).
- **Conclusion: INT4 quantization (W4 plain or W4+DCT) causes no accuracy degradation on LIBERO-Spatial.**
- Orin bf16 (83%) is within the expected range of A100 (~88%) — no edge-hardware degradation.

(Pilot n=15 numbers: bf16=93.3%, W4+DCT=100% — superseded by n=100 results above.)

## Accuracy — LIBERO spatial (success rate) — SUPERSEDED (fork-corrupted, see top)

| Model | Machine | Eval protocol | Precision | Success | n | Notes |
|---|---|---|---|---|---|---|
| ours 1-cam | A100 | 600 flat steps (old) | bf16 | 90.25% | — | inflated step budget |
| ours 1-cam | A100 | OFT 220/280/300/520 (std eval, center-crop) | bf16 | 88.2% | — | corrected; ~2pt below 600-step → steps barely matter |
| ours 1-cam | Orin | `07_` OFT 220, **no center-crop** | bf16 | 60.2% | 500 | eval-path bug: missing center-crop |
| ours 1-cam | Orin | `07_` OFT 220, **no center-crop** | INT4 (W4 fake-quant) | 56.0% | 100 | within noise of bf16-no-crop |
| ours 1-cam | Orin | `07_` OFT 220, **+center-crop** | bf16 | 53.3% | 15 | center-crop RULED OUT — same 8/15 as no-crop |
| ours 1-cam | Orin | `07_` no-crop | INT4 (W4 plain) | _running_ | 100 | sweep row 2 |
| ours 1-cam | Orin | `07_` no-crop | INT4 (W4 + DCT rotation) | _running_ | 100 | sweep row 3 (QuaRot/SpinQuant-style) |
| reference 2-cam | A100 | OFT (broken run) | bf16 | 46% 🔴 | — | broken eval, ignore |

**Read:** ours is ~88–90% on A100 either step budget. The Orin 60.2% gap is an eval-side issue but **center-crop is RULED OUT** (53.3% with vs 53.3% without on the same 3 tasks; per-task just reshuffled) — as was step-count. The ~28pt gap is still unexplained; need the EXACT A100 eval command (prompt format, normalization q01/q99 vs mean/std, image flip) to diff against `07_`. The quant sweep below measures INT4/DCT deltas RELATIVE to the Jetson bf16 baseline, which is valid independent of that mystery.

## Latency / Memory — Jetson AGX Orin (sm_87)

| Config | Backbone mem | Full-VLA peak | Query latency | Throughput |
|---|---|---|---|---|
| bf16 (PyTorch, full VLA) | ~12.5 GiB | 15.8 GiB | 320 ms/query | 3.1 q/s |
| INT4 GGUF Q4_K_M (backbone, llama.cpp CUDA) | **3.80 GiB** | ~6–7 GiB | prefill compute-bound → ~comparable | 903 t/s prefill, 28 t/s decode |

**Read:** INT4 = ~3.3× memory win, not a latency win (OFT query is one prefill pass = compute-bound).

## Open items
- [ ] Confirm center-crop recovers Orin bf16 → ~85–88% (A/B running).
- [ ] Rerun INT4 (W4) on Orin **with center-crop** for the corrected quant-cost delta.
- [ ] Confirm which checkpoint produced A100 88–90% has same norm-stats handling as `07_`.
