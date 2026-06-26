#!/usr/bin/env bash
set -euo pipefail

# Conservative OpenVLA LoRA fine-tuning launcher.
#
# Expected dataset layout for the default Bridge run:
#   $DATA_ROOT_DIR/bridge_dataset/<tfds files>
#
# You can override any value at launch, for example:
#   DATA_ROOT_DIR=/data/open-x DATASET_NAME=bridge_dataset MAX_STEPS=1000 ./train_openvla.sh

export PYTHONPATH="${PWD}/openvla:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-datasets/open-x-embodiment}"
DATASET_NAME="${DATASET_NAME:-bridge_dataset}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-checkpoints/bridge_lora}"
ADAPTER_TMP_DIR="${ADAPTER_TMP_DIR:-checkpoints/bridge_lora_adapters}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-16}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-$MAX_STEPS}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LORA_RANK="${LORA_RANK:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
IMAGE_AUG="${IMAGE_AUG:-False}"
USE_QUANTIZATION="${USE_QUANTIZATION:-True}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"

if [[ ! -d "${DATA_ROOT_DIR}/${DATASET_NAME}" ]]; then
  echo "Dataset directory not found: ${DATA_ROOT_DIR}/${DATASET_NAME}" >&2
  echo "Download/convert the RLDS dataset first, or set DATA_ROOT_DIR and DATASET_NAME." >&2
  exit 1
fi

torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" openvla/vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --adapter_tmp_dir "${ADAPTER_TMP_DIR}" \
  --use_lora True \
  --lora_rank "${LORA_RANK}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --use_quantization "${USE_QUANTIZATION}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --image_aug "${IMAGE_AUG}" \
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
