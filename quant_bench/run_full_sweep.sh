#!/usr/bin/env bash
# Full corrected-stack quant sweep (02_eval_libero.py, stock tf 4.40.2 + mujoco 3.9.0).
# libero_spatial, 10 tasks x 10 rollouts = n=100 per config. Three configs, sequential.
set -u
cd /home/agxorin/Desktop/offline-robotics
source venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CKPT=quant_bench/checkpoints/openvla_oft_libero/epoch_003
OUT=quant_bench/results
mkdir -p "$OUT"

run() { # name  quant_bits  rotation
  local name=$1 bits=$2 rot=$3
  echo "######## $name (bits=$bits rotation=$rot) @ $(date) ########"
  python quant_bench/02_eval_libero.py \
    --checkpoint $CKPT/hf_model --action_head $CKPT/action_head.pt --action_stats $CKPT/action_stats.json \
    --suite libero_spatial --rollouts_per_task 10 \
    --quant_bits $bits --rotation $rot \
    --output_csv $OUT/sweep_${name}.csv > $OUT/sweep_${name}.log 2>&1
  echo "$name EXIT=$? @ $(date)"
  tail -n 3 $OUT/sweep_${name}.log
}

run bf16      0 none
run w4_plain  4 none
run w4_dct    4 dct
echo "SWEEP DONE @ $(date)"
