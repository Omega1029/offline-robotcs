#!/usr/bin/env bash
# Transfer the latency/memory benchmark + checkpoint to the Jetson AGX Orin.
# RUN THIS ON THE A100 BOX. ~15.3 GB total (checkpoint dominates).
#
# PREREQ — accept the host key ONCE first (verify the fingerprint!):
#     ssh agxorin@10.10.10.112        # type 'yes' once, then exit
#
# Then:
#     bash transfer_to_jetson.sh
set -euo pipefail

JETSON=agxorin@10.10.10.112
# self-contained subdir under the existing repo, so we never clobber its files
DEST=/home/agxorin/Desktop/offline-robotics/quant_bench
HERE="$(cd "$(dirname "$0")" && pwd)"

# Connection multiplexing: authenticate ONCE (one password if no key), reuse for all calls.
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o ControlPath=/tmp/jetson_cm_%r@%h:%p -o ControlPersist=180"
SSH="ssh $SSH_OPTS"

echo ">>> 0/2  ensure destination exists"
$SSH "$JETSON" "mkdir -p $DEST"

echo ">>> 1/2  code bundle (~2 MB: scripts + prismatic source + HF remote-code cache)"
rsync -avhP -e "$SSH" "$HERE/jetson_bundle/" "$JETSON:$DEST/"

echo ">>> 2/2  checkpoint (~15.3 GB: hf_model + action_head.pt + action_stats.json)"
CKPT_REL="checkpoints/openvla_oft_libero/epoch_003"
$SSH "$JETSON" "mkdir -p $DEST/$CKPT_REL"
rsync -avhP -e "$SSH" \
  "$HERE/$CKPT_REL/hf_model" \
  "$HERE/$CKPT_REL/action_head.pt" \
  "$HERE/$CKPT_REL/action_stats.json" \
  "$JETSON:$DEST/$CKPT_REL/"

echo ">>> done. On the Jetson:"
echo "      cd $DEST && cat README_JETSON.md   # steps 1-3, then: bash run_bench.sh"
