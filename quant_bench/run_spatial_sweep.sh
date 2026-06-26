#!/usr/bin/env bash
# Full libero_spatial sweep on the corrected stack (stock transformers 4.40.2 + mujoco 3.9.0,
# 02_eval, causal/last-56). 10 tasks x N trials for bf16, W4+DCT, and W4-plain.
#   Usage: bash run_spatial_sweep.sh [N]   (N = rollouts per task, default 10)
set -euo pipefail
cd "$(dirname "$0")"

# Activate venv if present and not already active.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for v in ../venv ~/venv_quant; do
    if [[ -f "$v/bin/activate" ]]; then source "$v/bin/activate"; break; fi
  done
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl

CKPT=checkpoints/openvla_oft_libero/epoch_003/hf_model
STATS=checkpoints/openvla_oft_libero/epoch_003/action_stats.json
HEAD=checkpoints/openvla_oft_libero/epoch_003/action_head.pt
N="${1:-10}"

COMMON=(--checkpoint "$CKPT" --action_stats "$STATS" --action_head "$HEAD"
        --suite libero_spatial --rollouts_per_task "$N")

echo "############ 1) bf16 baseline (n=${N} x 10 tasks) ############"
python 02_eval_libero.py "${COMMON[@]}" \
  --output_csv "results/spatial_bf16_n${N}.csv"

echo "############ 2) W4 + DCT rotation (n=${N} x 10 tasks) ############"
python 02_eval_libero.py "${COMMON[@]}" \
  --quant_bits 4 --rotation dct \
  --output_csv "results/spatial_w4_dct_n${N}.csv"

echo "############ 3) W4 plain / no rotation (n=${N} x 10 tasks) ############"
python 02_eval_libero.py "${COMMON[@]}" \
  --quant_bits 4 --rotation none \
  --output_csv "results/spatial_w4_plain_n${N}.csv"

echo "############ done — summaries: ############"
for p in results/spatial_*_n${N}.summary.json; do
  [[ -f "$p" ]] && echo "$p" && cat "$p"
done
