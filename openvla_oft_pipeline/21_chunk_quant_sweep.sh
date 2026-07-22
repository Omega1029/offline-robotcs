#!/usr/bin/env bash
# Chunk-size (execution horizon) × quantization interaction sweep.
#
# Grid: execute_horizon h ∈ {1, 2, 4, 8} × quant ∈ {none, w4a4_dct, w3a8_dct}
#       (h = how many actions of each predicted 8-action chunk are executed
#        before re-querying; h=8 is the standard OFT open-loop chunk).
#
# Hypothesis: shorter horizons re-query more often — fresher visual feedback
# corrects accumulated quantization error (helps quant more than bf16), at the
# cost of h× more model queries (latency/energy on edge). The interaction
# term is the paper.
#
# Protocol: SCREEN on libero_spatial 10 tasks × 3 rollouts per cell (~30-40 min
# each sharing a GPU with training). Full 400-rollout evals on interesting
# cells afterwards. Resumable: cells with an existing summary JSON are skipped.
#
# USAGE:
#   bash 21_chunk_quant_sweep.sh                                   # default epoch_003
#   CKPT=checkpoints/openvla_oft_libero_2cam/epoch_004 bash 21_chunk_quant_sweep.sh

set -uo pipefail

CKPT="${CKPT:-checkpoints/openvla_oft_libero_2cam/epoch_003}"
ROLLOUTS="${ROLLOUTS:-3}"          # per task; 3 = screen, 10 = full
SUITE="${SUITE:-libero_spatial}"   # empty string = all 4 suites
OUTDIR="${OUTDIR:-results/chunk_quant_sweep}"
GPU="${GPU:-0}"
EPOCH_TAG=$(basename "${CKPT}")

mkdir -p "${OUTDIR}"

for QUANT in none w4a4_dct w3a8_dct; do
  for H in 8 4 2 1; do
    CELL="${EPOCH_TAG}_${QUANT}_h${H}"
    CSV="${OUTDIR}/${CELL}.csv"
    SUMMARY="${OUTDIR}/${CELL}.summary.json"
    if [[ -f "${SUMMARY}" ]]; then
      echo "[sweep] SKIP ${CELL} (summary exists)"
      continue
    fi
    echo "[sweep] RUN ${CELL}"
    MUJOCO_GL=egl PYTHONPATH="$(pwd)/openvla-oft" \
    ../venv/bin/python 02B_eval_libero_2cam.py \
      --checkpoint "${CKPT}/hf_model" \
      --action_stats "${CKPT}/action_stats.json" \
      ${SUITE:+--suite "${SUITE}"} \
      --rollouts_per_task "${ROLLOUTS}" \
      --execute_horizon "${H}" \
      --quant "${QUANT}" \
      --gpu "${GPU}" \
      --output_csv "${CSV}" \
      2>&1 | tail -6
    echo "[sweep] DONE ${CELL}"
  done
done

echo ""
echo "[sweep] All cells complete. Summary:"
for f in "${OUTDIR}"/${EPOCH_TAG}_*.summary.json; do
  python3 -c "
import json,sys
d = json.load(open('$f'))
srs = d['suite_success_rates']
print(f\"  quant={d['quant']:9s} h={d['execute_horizon']}: \" +
      ' '.join(f'{k.split(\"_\")[1]}={v:.0%}' for k,v in srs.items()) +
      f\"  avg={d['average_success_rate']:.1%}\")
"
done
