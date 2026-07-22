# Chunk-Size (Execution Horizon) × Quantization Interaction Study — Results

**Date:** 2026-07-16
**Model:** `checkpoints/openvla_oft_libero_2cam/epoch_003` (2-cam + proprio OFT, 95.5% all-suite bf16)
**Protocol:** LIBERO-Spatial, official 220-step budget. Screen = 10 tasks × 3 rollouts;
confirmation = 10 × 10 (±4% binomial CI). `02B_eval_libero_2cam.py --execute_horizon h --quant q`.
h = number of actions executed from each predicted 8-action chunk before re-querying.
Quant = DCT-rotation fake-quant of all 224 backbone linears (`quant_expirements/Transform`).

## Headline grid — full 10×10 confirmation

| Spatial success | h=8 | h=4 | h=2 | h=1 |
|---|---|---|---|---|
| bf16            | 88.0% | 83.0% | 75.0% | 60.0% |
| W4A4+DCT        | 90.0% | 82.0% | 74.0% | 63.0% |
| Δ (quant−bf16)  | +2.0  | −1.0  | −1.0  | +3.0 |
| queries/rollout | 15.5  | 33.2  | 72.3  | 162.7 |
| env steps/rollout | 130.4 | 141.3 | 154.2 | 172.7 |

Screen grid (10×3, incl. W3A8+DCT) in `results/chunk_quant_sweep/`; W3A8 tracked bf16
within noise at every h (77/80/70/60) — no need for full confirmation.

## Findings

1. **Execution horizon dominates; the collapse is large and monotone.**
   28 pts (bf16) / 27 pts (W4A4) lost from h=8 → h=1 — 7× the CI. Re-querying the
   policy every step *destroys* performance. Mechanism consistent with training
   distribution: the model predicts chunks from chunk-boundary states; querying
   mid-motion feeds it out-of-distribution proprio (arm mid-swing, gripper mid-close)
   and induces re-planning jitter. Corroborated by trajectory length: h=1 rollouts
   need 32% more env steps (172.7 vs 130.4) — the policy dithers.

2. **No precision × horizon interaction.** |Δ| ≤ 3 pts at every horizon (mean +0.75);
   the two curves fall in lockstep. Fresh visual feedback does NOT preferentially
   rescue quantization error (hypothesis falsified), and quantized policies do NOT
   suffer more from open-loop chunk execution. Precision and horizon are independent
   deployment knobs.

3. **W4A4+DCT is free at every horizon** — including the headline cell:
   90.0% vs 88.0% bf16 at h=8 (102.3% retention, statistically indistinguishable).
   Reproduces the mono-cam finding on the stronger 2-cam checkpoint.

4. **The accuracy-optimal horizon is also the compute-optimal one.** h=8 gives the
   best success at 10.5× fewer queries than h=1. On Jetson AGX Orin (330 ms/query
   real-kernel measurement), h=1 additionally drops the control rate below 3 Hz —
   infeasible regardless of accuracy. **Deployment rule: quantize W4A4+DCT, always
   execute full chunks.** There is no tradeoff to navigate.

## Paper framing (CoRL 26 workshop, 4 pages)

"Chunked execution is load-bearing, quantization is free: horizon × precision
interactions in edge VLA deployment." Figure 1 = the two overlapping falling curves
with queries/rollout on a second axis. Table 1 = grid above. Sec. 4 = Orin
latency/energy projection per cell (queries/rollout × J/query from tegrastats —
MEASUREMENT PENDING, the one remaining experiment).

## Raw data

- `results/chunk_quant_sweep/` — 12-cell screen (3 precisions × 4 horizons), CSVs + summaries
- `results/chunk_quant_full/` — 8-cell full confirmation (bf16, W4A4 × 4 horizons)
- Full 400-rollout all-suite baseline of this checkpoint: `results/full_2cam_epoch003.csv` (95.5%)
