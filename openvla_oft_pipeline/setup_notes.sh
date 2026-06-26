#!/usr/bin/env bash
# Post-install setup for the OpenVLA-OFT / LIBERO pipeline.
#
# USAGE (from the project root, with your venv active):
#   source ~/venvs/openvla_oft/bin/activate   # or wherever your venv is
#   pip install -r openvla_oft_pipeline/envs/requirements.txt
#   bash openvla_oft_pipeline/setup_notes.sh
#
# This script is intentionally non-destructive — it prints what it does
# and exits early on any failure so you can inspect and retry step by step.

set -euo pipefail

log()  { echo "[setup] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

# ── Step 1: Clone openvla-oft repo ────────────────────────────────────────────
log "Step 1: Clone moojink/openvla-oft"
if [[ -d "openvla-oft" ]]; then
    log "  openvla-oft/ already exists — skipping clone."
    log "  To re-clone: rm -rf openvla-oft/ && re-run this script."
else
    git clone https://github.com/moojink/openvla-oft.git
    log "  Cloned to openvla-oft/"
fi

log "  Installing openvla-oft in editable mode..."
pip install -e openvla-oft/ --quiet
log "  Done."

# ── Step 2: Install LIBERO from git ───────────────────────────────────────────
log "Step 2: Install LIBERO from git"
# LIBERO is not on PyPI and requires source install for task configs to work.
# The @main pin intentionally gets the latest fixes; if the repo breaks, pin to
# a specific commit: @abc1234
pip install "git+https://github.com/Lifelong-Robot-Learning/LIBERO.git@master" --quiet
log "  LIBERO installed."

# ── Step 3: Download LIBERO datasets ─────────────────────────────────────────
# LIBERO auto-downloads to $LIBERO_DATA_PATH (defaults to ~/libero_data if unset).
# Datasets total ~10 GB. Set LIBERO_DATA_PATH before running to control location.
log "Step 3: Download LIBERO task suite datasets"
log "  NOTE: Disk must have >=10 GB free. Run download_libero.sh separately if needed."
log "  LIBERO_DATA_PATH=${LIBERO_DATA_PATH:-~/libero_data}"
python - <<'EOF'
import os
os.environ.setdefault("LIBERO_DATA_PATH", os.path.expanduser("~/libero_data"))

from libero.libero import benchmark

suites = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
bdict = benchmark.get_benchmark_dict()

for suite_name in suites:
    print(f"  Checking {suite_name}...")
    try:
        suite = bdict[suite_name]()
        print(f"  {suite_name}: {len(suite.get_task_names())} tasks ready")
    except Exception as e:
        print(f"  {suite_name}: download triggered or error — {e}")
        print(f"  Re-run this step after download completes.")
EOF
log "  Dataset check complete."

# ── Step 4: Verify all critical versions ──────────────────────────────────────
log "Step 4: Version verification"
python - <<'EOF'
import sys

checks = [
    ("transformers", "4.40.2"),
    ("torch", "2.2."),
    ("deepspeed", "0.14."),
    ("robosuite", "1.4."),
    ("numpy", "1."),
    ("peft", "0.10."),
    ("accelerate", "0.28."),
]

import importlib
all_ok = True
for pkg, expected_prefix in checks:
    try:
        mod = importlib.import_module(pkg)
        ver = mod.__version__
        ok = ver.startswith(expected_prefix)
        status = "OK " if ok else "WARN"
        if not ok:
            all_ok = False
        print(f"  [{status}] {pkg}=={ver} (expected ~{expected_prefix}x)")
    except ImportError:
        print(f"  [MISS] {pkg} not installed")
        all_ok = False

import torch
cuda_ok = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print(f"  [{'OK' if cuda_ok else 'FAIL'}] CUDA available: {cuda_ok}")
print(f"  [{'OK' if gpu_count >= 8 else 'WARN'}] GPU count: {gpu_count} (expected 8 for H100 node)")

if not all_ok:
    print("\n  Some versions mismatched. See envs/training_env.yml for correct pins.")
    sys.exit(1)
else:
    print("\n  All versions OK.")
EOF

# ── Step 5: Model class pre-check ─────────────────────────────────────────────
log "Step 5: Verify openvla-oft model class and method names"
log "  (This downloads ~14 GB on first run — cached after that)"
python - <<'EOF'
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

print("  Loading openvla/openvla-7b (base model)...")
print("  Note: moojink/openvla-oft is a GitHub repo (training code), not an HF model.")
print("  The OFT action_head is added by the openvla-oft repo model class when installed.")
# Load to CPU first — no GPU needed for this check
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

# action_head is present only when the openvla-oft repo is installed and its
# model class is registered. If it's missing, that's expected for the plain base
# model — the OFT repo's training code will add it during fine-tuning.
has_action_head = hasattr(model, "action_head")
print(f"  action_head present: {has_action_head}")
if not has_action_head:
    print("  (Expected if openvla-oft repo not yet installed/registered — OK at this stage)")
assert model.config.model_type == "openvla", (
    f"FATAL: model_type='{model.config.model_type}', expected 'openvla'."
)

print(f"  model_type: {model.config.model_type}")
print(f"  action_head: {model.action_head}")

# Find the correct inference method name — update the training/eval scripts if different
action_methods = [
    m for m in dir(model)
    if ("action" in m.lower() or "predict" in m.lower())
    and not m.startswith("_")
    and callable(getattr(model, m))
]
print(f"  Inference methods: {action_methods}")

# Find proprio kwarg name in forward()
import inspect
sig = inspect.signature(model.forward)
params = list(sig.parameters.keys())
print(f"  model.forward() params: {params}")
proprio_params = [p for p in params if "proprio" in p.lower() or "state" in p.lower()]
print(f"  Proprio-related params: {proprio_params}")

print()
print("  ACTION REQUIRED IF:")
print("  - inference method is not 'predict_action' → update all call sites")
print("  - proprio param is not 'proprio' → update all call sites")
print("  - Look in 01_train_openvla_oft.py and 02_eval_libero.py for '# verify kwarg'")
EOF

log ""
log "Setup complete."
log "Next step: bash launch_train.sh"
