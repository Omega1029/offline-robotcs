#!/usr/bin/env bash
# Launch OpenVLA-OFT training on 8x H100 using DeepSpeed ZeRO-2.
#
# PREREQUISITES (run once):
#   1. conda env create -f envs/training_env.yml
#   2. conda activate openvla_oft_train
#   3. git clone https://github.com/moojink/openvla-oft.git
#      pip install -e openvla-oft/
#   4. pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git@master
#   5. Set DATA_ROOT below to your LIBERO dataset directory
#   6. Set WANDB_API_KEY or run: wandb login
#
# VERSION CHECK — run before first launch:
#   python -c "
#   import transformers, torch, deepspeed, robosuite
#   print('transformers:', transformers.__version__)   # must be 4.40.2
#   print('torch:', torch.__version__)                 # must match your CUDA
#   print('deepspeed:', deepspeed.__version__)         # must be 0.14.2
#   print('robosuite:', robosuite.__version__)         # must be 1.4.1
#   print('cuda available:', torch.cuda.is_available())
#   print('gpu count:', torch.cuda.device_count())     # expect 8
#   "
#
# MODEL CLASS VERIFICATION — run before first launch:
#   python -c "
#   from transformers import AutoModelForVision2Seq, AutoProcessor
#   m = AutoModelForVision2Seq.from_pretrained('moojink/openvla-oft', trust_remote_code=True)
#   assert hasattr(m, 'action_head'), 'action_head missing — wrong model class!'
#   print('action_head:', m.action_head)
#   print('model_type:', m.config.model_type)  # expect: openvla
#   # Also check the predict_action method name:
#   action_methods = [x for x in dir(m) if 'action' in x.lower() or 'predict' in x.lower()]
#   print('action-related methods:', action_methods)
#   "
#   If the method name differs from 'predict_action', update every call site in
#   01_train_openvla_oft.py and 02_eval_libero.py before training.
#
# USAGE:
#   bash launch_train.sh                     # default settings
#   DATA_ROOT=/nvme/libero bash launch_train.sh
#   PER_GPU_BATCH=16 EPOCHS=15 bash launch_train.sh

set -euo pipefail

# ── Required: set your LIBERO dataset path ────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/home/justin_williams1/OfflineAutonomy/offline-robotcs/datasets/libero}"
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: DATA_ROOT=${DATA_ROOT} does not exist."
    echo "Set DATA_ROOT to the directory containing libero_spatial/, libero_object/, etc."
    exit 1
fi

# ── Configuration ─────────────────────────────────────────────────────────────
# moojink/openvla-oft is a GitHub repo (training code), not an HF model.
# Base model is always openvla/openvla-7b. The OFT action_head is added by
# the installed openvla-oft repo's model class (active when PYTHONPATH is set).
MODEL_ID="${MODEL_ID:-openvla/openvla-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/openvla_oft_libero}"
RUN_NAME="${RUN_NAME:-openvla_oft_libero_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-1}"                 # short run by default; use EPOCHS=10 MAX_STEPS_PER_EPOCH=0 for full training
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-2500}"  # ~2500 steps ≈ 30-35 min; 0 = full 6.7 h epoch
PER_GPU_BATCH="${PER_GPU_BATCH:-4}"   # reduced from 8 to ease GPU memory pressure
GRAD_ACCUM="${GRAD_ACCUM:-4}"         # doubled to keep effective batch = 4×4×8 = 128
LR="${LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
EVAL_EVERY="${EVAL_EVERY:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"       # reduced from 4 to cut HDF5 file contention
NPROC="${NPROC:-4}"                   # default 4 — other GPUs stay free for colleagues
GPUS="${GPUS:-}"                      # e.g. GPUS=4,5,6,7 to pick specific GPUs; empty = first NPROC
INIT_WEIGHTS="${INIT_WEIGHTS:-}"      # warm-start weights (recover a crashed run); empty = train from base
START_EPOCH="${START_EPOCH:-0}"       # first epoch index to train (skip completed epochs on resume)
KEEP_LAST_N="${KEEP_LAST_N:-1}"       # keep heavy optimizer state for only the last N epochs

# Optimizer state for full 7.69B-param fine-tuning is ~123 GB total — ZeRO-2 only
# shards it across GPUs, so with 1-2 GPUs it exceeds 80 GB/GPU and AdamW state
# must offload to CPU RAM (~2-3x slower steps). 3+ GPUs fit without offload.
if [[ -z "${DS_CONFIG:-}" && "${NPROC}" -lt 3 ]]; then
    DS_CONFIG="configs/ds_zero2_offload.json"
else
    DS_CONFIG="${DS_CONFIG:-configs/ds_zero2.json}"
fi

# ── Environment ───────────────────────────────────────────────────────────────
if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
fi
export MUJOCO_GL=egl           # headless rendering — osmesa is 3x slower
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export DEEPSPEED_TIMEOUT=60    # minutes; collective-op timeout (default 30, watchdog saw 10)
export PYTHONPATH="$(pwd)/openvla-oft:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"  # use 'online' and set WANDB_API_KEY to sync

# Effective batch size: PER_GPU_BATCH × GRAD_ACCUM × NPROC samples per optimizer step
echo "Effective batch size: $((PER_GPU_BATCH * GRAD_ACCUM * NPROC)) (${PER_GPU_BATCH} × ${GRAD_ACCUM} accum × ${NPROC} GPUs)  [ds_config: ${DS_CONFIG}]"

# ── torchrun + DeepSpeed launcher ─────────────────────────────────────────────
# Using torchrun (not deepspeed launcher) for better compatibility with DeepSpeed 0.14+
# torchrun injects LOCAL_RANK / WORLD_SIZE / RANK automatically
torchrun \
    --nnodes 1 \
    --nproc-per-node "${NPROC}" \
    --master_port 29500 \
    01_train_openvla_oft.py \
    --model_id "${MODEL_ID}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --ds_config "${DS_CONFIG}" \
    --num_epochs "${EPOCHS}" \
    --per_gpu_batch_size "${PER_GPU_BATCH}" \
    --grad_accum "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --eval_every "${EVAL_EVERY}" \
    --num_workers "${NUM_WORKERS}" \
    --max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}" \
    --start_epoch "${START_EPOCH}" \
    --keep_last_n "${KEEP_LAST_N}" \
    ${INIT_WEIGHTS:+--init_weights "${INIT_WEIGHTS}"} \
    --augment

echo ""
echo "Training complete. Checkpoint in: ${OUTPUT_DIR}"
echo "Run evaluation with:"
echo "  MUJOCO_GL=egl python 02_eval_libero.py \\"
echo "    --checkpoint ${OUTPUT_DIR}/epoch_$(printf '%03d' ${EPOCHS})/hf_model \\"
echo "    --action_stats ${OUTPUT_DIR}/epoch_$(printf '%03d' ${EPOCHS})/action_stats.json \\"
echo "    --output_csv results/libero_eval_final.csv \\"
echo "    --rollouts_per_task 10"
