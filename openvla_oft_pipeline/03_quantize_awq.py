"""
Post-training quantization of OpenVLA-OFT using AWQ INT8.

Strategy:
    The OFT model has three components with different quantization sensitivities:
    1. LLM backbone (Llama ~7B): largest, quantize with AWQ for memory savings
    2. Vision encoder (ViT): kept at bfloat16 — small and sensitive to precision
    3. Action head (MLP regressor): kept at bfloat16 — critical path for action accuracy

    We use AutoAWQ with modules_to_not_convert to exclude the vision tower and action head,
    then validate that action output shape and range are preserved.

    Test INT8 first. If success rate drops >3% vs fp16, try increasing calibration
    samples (--num_calib_samples 512) before moving to INT4.

USAGE:
    # INT8 (recommended first step)
    python 03_quantize_awq.py \
        --checkpoint checkpoints/openvla_oft_libero/epoch_003/hf_model \
        --action_stats checkpoints/openvla_oft_libero/epoch_003/action_stats.json \
        --data_root $DATA_ROOT \
        --output_dir checkpoints/openvla_oft_libero_awq_int8 \
        --bits 8

    # INT4 (smaller, use only if INT8 validated first)
    python 03_quantize_awq.py \
        --checkpoint checkpoints/openvla_oft_libero/epoch_010/hf_model \
        --action_stats checkpoints/openvla_oft_libero/epoch_010/action_stats.json \
        --data_root /path/to/libero_datasets \
        --output_dir checkpoints/openvla_oft_libero_awq_int4 \
        --bits 4

DEFAULT BACKEND (--method bnb):
    AutoAWQ dispatches by architecture name and does NOT recognize the custom
    OpenVLAForActionPrediction (Vision2Seq) class, so --method awq fails at
    from_pretrained. The default path uses bitsandbytes, which swaps nn.Linear
    layers during the generic from_pretrained (no architecture registry involved),
    quantizing the inner Llama backbone while skipping vision_backbone, projector,
    and lm_head. The action head is attached post-load and stays bf16. bnb is
    data-free, so --num_calib_samples / --group_size are ignored for this backend.

        # INT8 (recommended)
        python 03_quantize_awq.py --method bnb --bits 8 \
            --checkpoint .../hf_model --action_stats .../action_stats.json \
            --data_root $DATA_ROOT --output_dir checkpoints/oft_bnb_int8

        # NF4 4-bit (use only if INT8 validated first)
        python 03_quantize_awq.py --method bnb --bits 4 ...

KNOWN ISSUES:
    - AutoAWQ < 0.2.0 does not support modules_to_not_convert reliably. Stay at 0.2.5.
    - --method awq currently raises at from_pretrained: the OFT custom model class is
      not in AutoAWQ's registry. A manual LLM-only AWQ fallback (quantize the inner
      Llama standalone, graft back) is not yet implemented — use --method bnb.
    - The earlier claim that bnb is incompatible with the OFT action head is wrong:
      the action head is added after load and is not quantized, and the vision tower
      / projector are excluded via llm_int8_skip_modules.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Make the vendored openvla-oft tree importable so `prismatic` (action head) resolves.
# Mirrors 02_eval_libero.py.
_OPENVLA_OFT_DIR = Path(__file__).resolve().parent / "openvla-oft"
if _OPENVLA_OFT_DIR.is_dir() and str(_OPENVLA_OFT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPENVLA_OFT_DIR))

import h5py
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
OFT_CHUNK_SIZE = 8
OFT_ACTION_DIM = 7
OFT_ACTION_LEN = OFT_CHUNK_SIZE * OFT_ACTION_DIM  # = 56 appended action-token positions

# Vicuna-v1.5 prompt format used by openvla. Byte-identical to training
# (01_train_openvla_oft.py) and eval (02_eval_libero.py) — the prefix must match
# exactly or the model is off-distribution.
_SYS_MSG = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)


def _build_prompt(instruction: str) -> str:
    return f"{_SYS_MSG}USER: What action should the robot take to {instruction}? ASSISTANT: "


def _read_instruction(f) -> str:
    """
    The language instruction lives in f["data"].attrs["problem_info"] as a JSON string
    under "language_instruction" — NOT in the root f.attrs. Mirrors 01_train_openvla_oft.py.
    """
    problem_info = f["data"].attrs["problem_info"]
    if isinstance(problem_info, bytes):
        problem_info = problem_info.decode()
    return json.loads(problem_info)["language_instruction"]


def _read_agentview(f, demo_key: str, t: int) -> Image.Image:
    """
    Read agentview_rgb and apply the 180° flip ([::-1, ::-1]) that both training and
    eval apply. Skipping the flip silently puts the image upside-down vs training.
    """
    image_np = f["data"][demo_key]["obs"]["agentview_rgb"][t][::-1, ::-1]
    return Image.fromarray(np.ascontiguousarray(image_np))


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data

def load_calibration_samples(
    data_root: str,
    processor,
    num_samples: int = 128,
    suite_names: list[str] | None = None,
) -> list[dict]:
    """
    Load calibration samples from LIBERO trajectories.

    AWQ calibration must use DOMAIN-SPECIFIC data (LIBERO frames + instructions),
    NOT generic text prompts. Using generic text severely underestimates activation
    magnitudes for the visual tokens, leading to poor quantization quality.

    Returns list of input dicts ready to pass to model.forward().
    """
    if suite_names is None:
        suite_names = LIBERO_SUITES

    samples = []
    per_suite = max(1, num_samples // len(suite_names))

    for suite_name in suite_names:
        suite_dir = Path(data_root) / suite_name
        hdf5_files = sorted(suite_dir.glob("*.hdf5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files in {suite_dir}")

        suite_samples = 0
        for hdf5_path in hdf5_files:
            if suite_samples >= per_suite:
                break
            with h5py.File(hdf5_path, "r") as f:
                instruction = _read_instruction(f)
                for demo_key in list(f["data"].keys()):
                    if suite_samples >= per_suite:
                        break
                    T = f["data"][demo_key]["actions"].shape[0]
                    # Sample evenly across the demo — not just first frames
                    sample_indices = np.linspace(0, T - 1, min(5, T), dtype=int)
                    for t in sample_indices:
                        if suite_samples >= per_suite:
                            break
                        image_pil = _read_agentview(f, demo_key, t)
                        inputs = processor(
                            text=_build_prompt(instruction),
                            images=image_pil,
                            return_tensors="pt",
                        )
                        # Move to CPU for calibration data collection
                        samples.append({k: v for k, v in inputs.items()})
                        suite_samples += 1

    print(f"[Calibration] Loaded {len(samples)} samples from {len(suite_names)} suites")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Action output validation

def validate_action_output(
    model,
    processor,
    action_stats: dict,
    data_root: str,
    device: torch.device,
    label: str = "model",
    num_validation_samples: int = 20,
) -> dict:
    """
    Validate that action outputs from a model have the correct shape and reasonable range.

    Quantization can silently change output distributions without raising errors.
    This catches:
    1. Shape mismatches (wrong chunk_size or action_dim)
    2. Collapsed outputs (all-zero or constant actions — model is broken)
    3. Out-of-range outputs (mean |action| > 5σ from calibration stats)

    Returns dict with validation metrics. Raises AssertionError on hard failures.
    """
    action_mean = np.array(action_stats["mean"])
    action_std = np.array(action_stats["std"])

    # The OFT checkpoint resolves to the original discrete-token openvla-7b remote code,
    # which has NO predict_action method and no proprio input. Inference mirrors training:
    # run a raw forward with output_hidden_states, read the appended action-token positions,
    # and map them through the L1RegressionActionHead. Requires model.eval_action_token_ids,
    # which attach_action_head() sets up. See 02_eval_libero.py:run_rollout.
    if not hasattr(model, "eval_action_token_ids"):
        raise AttributeError(
            "model.eval_action_token_ids is not set — call attach_action_head() before "
            "validate_action_output()."
        )
    action_token_ids = model.eval_action_token_ids.to(device)

    suite_dir = Path(data_root) / LIBERO_SUITES[0]
    hdf5_path = next(suite_dir.glob("*.hdf5"))

    all_preds = []
    model.eval()

    with h5py.File(hdf5_path, "r") as f:
        instruction = _read_instruction(f)
        prefix_ids = processor.tokenizer(
            _build_prompt(instruction), add_special_tokens=True, return_tensors="pt"
        )["input_ids"].squeeze(0).to(device)
        input_ids = torch.cat([prefix_ids, action_token_ids]).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        demo_keys = list(f["data"].keys())[:num_validation_samples]
        for demo_key in demo_keys:
            t = f["data"][demo_key]["actions"].shape[0] // 2
            image_pil = _read_agentview(f, demo_key, t)
            pixel_values = processor.image_processor(
                images=image_pil, return_tensors="pt"
            )["pixel_values"].to(device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                )
                # Last OFT_ACTION_LEN positions are the action tokens (no EOS at inference).
                action_hidden = outputs.hidden_states[-1][:, -OFT_ACTION_LEN:, :]
                # (1, chunk_size, action_dim) in mean/std-normalized space.
                pred = model.action_head.predict_action(action_hidden)

            all_preds.append(pred.float().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, chunk_size, action_dim)

    # Hard shape check
    assert all_preds.shape[1] == OFT_CHUNK_SIZE, (
        f"[{label}] Action chunk size mismatch: got {all_preds.shape[1]}, "
        f"expected {OFT_CHUNK_SIZE}. Quantization broke the action head."
    )
    assert all_preds.shape[2] == OFT_ACTION_DIM, (
        f"[{label}] Action dim mismatch: got {all_preds.shape[2]}, "
        f"expected {OFT_ACTION_DIM}."
    )

    # Unnormalize for range check
    unnorm = all_preds * action_std[None, None, :] + action_mean[None, None, :]

    # Collapse check: std across samples should be > 0.01 per dim
    pred_std = unnorm.reshape(-1, OFT_ACTION_DIM).std(axis=0)
    if np.any(pred_std < 0.005):
        low_dims = np.where(pred_std < 0.005)[0].tolist()
        print(f"[{label}] WARNING: Near-constant action output on dims {low_dims}. "
              f"Stds: {pred_std.tolist()}. Model may be collapsed after quantization.")

    # Range check against training distribution
    unnorm_flat = unnorm.reshape(-1, OFT_ACTION_DIM)
    mean_deviation = np.abs(unnorm_flat.mean(axis=0) - action_mean) / (action_std + 1e-8)
    if np.any(mean_deviation > 3.0):
        flagged = np.where(mean_deviation > 3.0)[0].tolist()
        print(f"[{label}] WARNING: Action mean shifted >3σ from training dist on dims {flagged}. "
              f"Consider more calibration samples or use INT8 instead of INT4.")

    metrics = {
        "shape": list(all_preds.shape),
        "unnorm_mean": unnorm_flat.mean(axis=0).tolist(),
        "unnorm_std": pred_std.tolist(),
        "mean_deviation_sigma": mean_deviation.tolist(),
    }

    print(f"[{label}] Validation passed. Shape: {all_preds.shape}")
    print(f"[{label}] Action mean deviation from training: {mean_deviation.round(3).tolist()} σ")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Action head (re)construction
#
# The OFT action_head is NOT part of the HF checkpoint: save_pretrained() only
# serializes the VLM. Training attaches an L1RegressionActionHead at runtime and
# saves it separately to action_head.pt. Both the fp16 baseline and the quantized
# model must reconstruct it here and attach it as model.action_head — mirroring
# load_model() in 02_eval_libero.py.

def attach_action_head(model, processor, action_head_path: str, device, dtype=torch.bfloat16):
    """Reconstruct L1RegressionActionHead, load action_head.pt, attach to model.

    Also builds model.eval_action_token_ids (placeholder action tokens appended to the
    prompt) so validate_action_output can read the action-token hidden states.
    """
    from prismatic.models.action_heads import L1RegressionActionHead

    llm_dim = model.language_model.config.hidden_size  # 4096 for LLaMA-2-7B
    action_head = L1RegressionActionHead(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=OFT_ACTION_DIM,
    ).to(device, dtype=dtype)

    if not Path(action_head_path).exists():
        raise FileNotFoundError(
            f"action_head weights not found at {action_head_path}. Training saves these "
            f"as action_head.pt next to (or one level above) the hf_model/ checkpoint — "
            f"pass --action_head explicitly if they live elsewhere."
        )
    state_dict = torch.load(action_head_path, map_location=device)
    action_head.load_state_dict(state_dict)
    action_head.eval()
    model.action_head = action_head

    # Build the placeholder action-token IDs consumed by the forward path in
    # validate_action_output. These must match 01_train_openvla_oft.py EXACTLY: build
    # the IDs directly (no ID→text→ID round-trip — that was not reversible for these
    # special tokens and misaligned the -56: slice). Mirrors 02_eval_libero.py:load_model.
    from prismatic.vla.action_tokenizer import ActionTokenizer

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    zero_chunk = np.zeros((OFT_CHUNK_SIZE, OFT_ACTION_DIM), dtype=np.float32)
    disc = np.digitize(np.clip(zero_chunk, -1.0, 1.0), action_tokenizer.bins)
    action_token_ids = torch.from_numpy(
        (processor.tokenizer.vocab_size - disc).reshape(-1).astype(np.int64)
    )
    assert action_token_ids.numel() == OFT_ACTION_LEN
    model.eval_action_token_ids = action_token_ids

    return action_head


# ─────────────────────────────────────────────────────────────────────────────
# AWQ quantization

def quantize_with_awq(
    checkpoint_path: str,
    action_head_path: str,
    output_dir: str,
    data_root: str,
    action_stats: dict,
    bits: int = 8,
    num_calib_samples: int = 128,
    group_size: int = 128,
):
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError(
            "autoawq is not installed. Run: pip install autoawq==0.2.5 autoawq-kernels==0.0.6"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("AWQ quantization requires a GPU. Use a machine with CUDA.")

    print(f"[AWQ] Loading processor from {checkpoint_path}")
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    # ── Baseline validation (fp16) ────────────────────────────────────────────
    print("[AWQ] Running pre-quantization baseline validation...")
    baseline_model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # The action head is never in the HF checkpoint — reconstruct and attach it.
    print(f"[AWQ] Attaching fp16 action head from {action_head_path}")
    attach_action_head(baseline_model, processor, action_head_path, device, dtype=torch.bfloat16)

    baseline_metrics = validate_action_output(
        baseline_model, processor, action_stats, data_root, device, label="fp16_baseline"
    )

    # Save action head weights separately — we re-attach them after quantization
    # so the action head stays in fp16
    print("[AWQ] Extracting action head weights (will be kept at fp16)...")
    action_head_state = {
        k: v.clone().cpu()
        for k, v in baseline_model.action_head.state_dict().items()
    }
    del baseline_model
    torch.cuda.empty_cache()

    # ── AWQ quantization ──────────────────────────────────────────────────────
    print(f"[AWQ] Loading model for INT{bits} quantization via AutoAWQ...")

    # modules_to_not_convert: exclude vision tower and action head from quantization.
    # These module name patterns must match the actual attribute names in the OFT model.
    # Inspect with: list(model.named_modules())[:20]
    # Common patterns in prismatic/OpenVLA-style models:
    MODULES_TO_SKIP = [
        "vision_backbone",   # ViT image encoder
        "vision_tower",      # alternative name
        "action_head",       # OFT regression head — keep in fp16
        "embed_tokens",      # embedding table — AWQ doesn't quantize these anyway
        "lm_head",           # output projection
    ]

    try:
        awq_model = AutoAWQForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,
        )
    except Exception as e:
        print(f"[AWQ] AutoAWQForCausalLM.from_pretrained failed: {e}")
        print("[AWQ] The OFT model class may not be recognized by AutoAWQ.")
        print("[AWQ] Manual quantization fallback is not yet implemented in this script.")
        print("[AWQ] Options:")
        print("  1. File an issue at github.com/casper-hansen/AutoAWQ requesting OFT support.")
        print("  2. Use bitsandbytes INT8 for the backbone only (see inline comment).")
        raise

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",
        "modules_to_not_convert": MODULES_TO_SKIP,
    }

    print(f"[AWQ] Loading calibration data ({num_calib_samples} samples)...")
    calib_samples = load_calibration_samples(data_root, processor, num_samples=num_calib_samples)

    print(f"[AWQ] Running calibration and INT{bits} quantization (this takes ~30-90 min)...")
    awq_model.quantize(
        tokenizer=processor,
        quant_config=quant_config,
        calib_data=calib_samples,
    )

    # Re-attach fp16 action head. The quantized model has no action_head attribute
    # (it was never in the checkpoint), so reconstruct it and load the saved fp16 weights.
    print("[AWQ] Re-attaching fp16 action head to quantized model...")
    head = attach_action_head(awq_model.model, processor, action_head_path, device, dtype=torch.bfloat16)
    head.load_state_dict(action_head_state)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    awq_model.save_quantized(str(output_path))
    processor.save_pretrained(str(output_path))

    # Save quant config and action stats alongside the model
    with open(output_path / "quant_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)
    with open(output_path / "action_stats.json", "w") as f:
        json.dump(action_stats, f, indent=2)

    print(f"[AWQ] Quantized model saved to {output_path}")

    # ── Post-quantization validation ──────────────────────────────────────────
    print("[AWQ] Running post-quantization validation...")

    # Load the quantized model for validation
    from awq import AutoAWQForCausalLM as AWQLoader
    val_wrapper = AWQLoader.from_quantized(
        str(output_path),
        fuse_layers=True,     # fused GEMM kernels — used in deployment too
        trust_remote_code=True,
    ).to(device).eval()

    # The reloaded checkpoint has no action head (never serialized) and no
    # eval_action_token_ids — reconstruct both on the inner HF model, then validate it.
    val_model = val_wrapper.model
    val_head = attach_action_head(val_model, processor, action_head_path, device, dtype=torch.bfloat16)
    val_head.load_state_dict(action_head_state)

    quant_metrics = validate_action_output(
        val_model, processor, action_stats, data_root, device,
        label=f"awq_int{bits}"
    )

    # Compare to baseline
    base_means = np.array(baseline_metrics["unnorm_mean"])
    quant_means = np.array(quant_metrics["unnorm_mean"])
    shift_sigma = np.abs(quant_means - base_means) / (np.array(action_stats["std"]) + 1e-8)
    max_shift = shift_sigma.max()

    print(f"\n[AWQ] Validation Summary:")
    print(f"  INT{bits} vs fp16 max action mean shift: {max_shift:.3f}σ")
    if max_shift > 2.0:
        print(f"  WARNING: Large distribution shift ({max_shift:.1f}σ). "
              f"Consider: more calibration samples, larger group_size=64, or stay with INT8.")
    else:
        print(f"  PASSED: Distribution shift within acceptable range (<2σ).")

    print(f"\n  Disk size estimate:")
    total_size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e6
    print(f"  {output_path}: {total_size_mb:.0f} MB")

    return {
        "baseline": baseline_metrics,
        "quantized": quant_metrics,
        "max_shift_sigma": float(max_shift),
        "bits": bits,
        "output_dir": str(output_dir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# bitsandbytes quantization (LLM backbone INT8 / NF4)
#
# Why this path exists: AutoAWQ dispatches by architecture name and does not
# recognize the custom OpenVLAForActionPrediction (Vision2Seq) class, so the AWQ
# path above fails at from_pretrained. bitsandbytes swaps nn.Linear layers during
# the generic from_pretrained based on quantization_config — independent of any
# architecture registry — so it quantizes the inner Llama backbone fine while we
# skip the vision tower, projector, and lm_head. The action head is attached after
# load (never in the checkpoint) and stays bf16. Unlike AWQ, bnb is data-free:
# no calibration samples needed.

# Top-level submodules to keep at bf16 (substring match on module names).
# Verified against prismatic/extern/hf/modeling_prismatic.py:
#   self.vision_backbone, self.projector, self.language_model
# "lm_head" matches language_model.lm_head. The action head is added post-load.
BNB_SKIP_MODULES = ["vision_backbone", "projector", "lm_head"]


def _patch_dispatch_model_for_bnb():
    """Bridge a transformers-4.40 / accelerate-1.8 / bnb version mismatch.

    transformers 4.40.x calls accelerate.dispatch_model (modeling_utils.py:3735)
    BEFORE it sets the model's is_loaded_in_8bit / is_loaded_in_4bit flags
    (set later in postprocess_model, line 3738). accelerate >=1.x reads those flags
    at the START of dispatch_model to decide between attaching device hooks and its
    single-device "collapse" path. Because the flags aren't set yet, accelerate
    takes the collapse path, which runs `model.to(device)` — and bitsandbytes
    forbids that ("`.to` is not supported for `4-bit` or `8-bit` bnb models").
    (For 4-bit with bnb>=0.43.2 accelerate also deliberately skips force_hooks, so
    the collapse path is taken there regardless.)

    Fix: force accelerate onto its hook-dispatch path, which correctly places both
    quantized weights AND non-persistent buffers (e.g. rotary_emb.inv_freq) on the
    target device. We set is_loaded_in_8bit only for the duration of dispatch, then
    restore the prior state so transformers' postprocess records the truthful flags.
    Idempotent; only affects this process.
    """
    import transformers.modeling_utils as _mu

    if getattr(_mu, "_bnb_dispatch_patched", False):
        return
    _orig_dispatch = _mu.dispatch_model
    _MISSING = object()

    def _patched_dispatch(model, **kwargs):
        qcfg = getattr(getattr(model, "config", None), "quantization_config", None)
        is_bnb = qcfg is not None and (
            getattr(qcfg, "load_in_8bit", False) or getattr(qcfg, "load_in_4bit", False)
        )
        if not is_bnb:
            return _orig_dispatch(model, **kwargs)

        saved = getattr(model, "is_loaded_in_8bit", _MISSING)
        model.is_loaded_in_8bit = True  # force accelerate's hook path during dispatch
        try:
            return _orig_dispatch(model, **kwargs)
        finally:
            if saved is _MISSING:
                try:
                    delattr(model, "is_loaded_in_8bit")
                except AttributeError:
                    pass
            else:
                model.is_loaded_in_8bit = saved

    _mu.dispatch_model = _patched_dispatch
    _mu._bnb_dispatch_patched = True


def quantize_with_bnb(
    checkpoint_path: str,
    action_head_path: str,
    output_dir: str,
    data_root: str,
    action_stats: dict,
    bits: int = 8,
):
    from transformers import BitsAndBytesConfig

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("bitsandbytes quantization requires a GPU. Use a machine with CUDA.")

    print(f"[BNB] Loading processor from {checkpoint_path}")
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    # ── Baseline validation (bf16) ──────────────────────────────────────────────
    print("[BNB] Running pre-quantization baseline validation...")
    baseline_model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    print(f"[BNB] Attaching bf16 action head from {action_head_path}")
    attach_action_head(baseline_model, processor, action_head_path, device, dtype=torch.bfloat16)
    baseline_metrics = validate_action_output(
        baseline_model, processor, action_stats, data_root, device, label="bf16_baseline"
    )
    del baseline_model
    torch.cuda.empty_cache()

    # ── bitsandbytes quantization ───────────────────────────────────────────────
    if bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=BNB_SKIP_MODULES,
            # Keep the action head and any kept-precision Linear in their native dtype.
            llm_int8_threshold=6.0,
        )
    elif bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # llm_int8_skip_modules is the shared skip list — also honored in 4-bit.
            llm_int8_skip_modules=BNB_SKIP_MODULES,
        )
    else:
        raise ValueError(f"bnb supports bits in {{4, 8}}, got {bits}")

    print(f"[BNB] Loading model with INT{bits} quantization (data-free, no calibration)...")
    # bnb requires device placement at load time via device_map — a quantized model
    # cannot be moved with .to(device) afterwards. Patch dispatch_model first to work
    # around the transformers/accelerate version mismatch (see helper docstring).
    _patch_dispatch_model_for_bnb()
    quant_model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()

    # Action head is never serialized in the checkpoint — reconstruct and attach it
    # in bf16. It is created fresh (not quantized) so it stays full precision.
    print("[BNB] Attaching bf16 action head to quantized model...")
    attach_action_head(quant_model, processor, action_head_path, device, dtype=torch.bfloat16)

    # ── Save ────────────────────────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # save_pretrained serializes the quantized weights and embeds quantization_config
    # in config.json, so reload is automatic. trust_remote_code custom modeling files
    # (referenced via auto_map) are copied alongside.
    quant_model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))

    quant_config_dump = bnb_config.to_dict()
    with open(output_path / "quant_config.json", "w") as f:
        json.dump(quant_config_dump, f, indent=2, default=str)
    with open(output_path / "action_stats.json", "w") as f:
        json.dump(action_stats, f, indent=2)

    print(f"[BNB] Quantized model saved to {output_path}")

    # ── Post-quantization validation ────────────────────────────────────────────
    print("[BNB] Running post-quantization validation...")
    quant_metrics = validate_action_output(
        quant_model, processor, action_stats, data_root, device, label=f"bnb_int{bits}"
    )

    base_means = np.array(baseline_metrics["unnorm_mean"])
    quant_means = np.array(quant_metrics["unnorm_mean"])
    shift_sigma = np.abs(quant_means - base_means) / (np.array(action_stats["std"]) + 1e-8)
    max_shift = float(shift_sigma.max())

    print(f"\n[BNB] Validation Summary:")
    print(f"  INT{bits} vs bf16 max action mean shift: {max_shift:.3f}σ")
    if max_shift > 2.0:
        print(f"  WARNING: Large distribution shift ({max_shift:.1f}σ). "
              f"INT4(NF4) can be lossy on the action path — try INT8 instead.")
    else:
        print(f"  PASSED: Distribution shift within acceptable range (<2σ).")

    total_size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e6
    print(f"\n  Disk size estimate:")
    print(f"  {output_path}: {total_size_mb:.0f} MB")

    return {
        "method": "bnb",
        "baseline": baseline_metrics,
        "quantized": quant_metrics,
        "max_shift_sigma": max_shift,
        "bits": bits,
        "output_dir": str(output_dir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to hf_model/ directory from training")
    p.add_argument("--action_head", default=None,
                   help="Path to action_head.pt (saved separately during training). "
                        "Defaults to action_head.pt next to (or one level above) --checkpoint.")
    p.add_argument("--action_stats", required=True,
                   help="Path to action_stats.json")
    p.add_argument("--data_root", required=True,
                   help="Root dir of LIBERO datasets (for calibration)")
    p.add_argument("--output_dir", required=True,
                   help="Where to save the quantized model")
    p.add_argument("--method", default="bnb", choices=["bnb", "awq"],
                   help="Quantization backend. 'bnb' (default): bitsandbytes INT8/NF4, "
                        "works with the custom OFT model class, data-free, no extra install. "
                        "'awq': AutoAWQ — currently fails on OFT (model class not in AWQ's "
                        "architecture registry); kept for when manual LLM-only AWQ is added.")
    p.add_argument("--bits", type=int, default=8, choices=[4, 8],
                   help="Quantization bit-width. Test INT8 before INT4.")
    p.add_argument("--num_calib_samples", type=int, default=128,
                   help="Number of LIBERO calibration samples. 128 is usually enough; "
                        "increase to 512 if INT4 validation shows large shift.")
    p.add_argument("--group_size", type=int, default=128,
                   help="AWQ group size. Smaller group = better accuracy, more overhead. "
                        "128 is standard; use 64 if INT4 quality is poor.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.action_stats) as f:
        action_stats = json.load(f)

    # Resolve action_head.pt: training writes it next to the epoch dir (one level
    # above hf_model/). Fall back to the checkpoint dir itself. Mirrors 02_eval_libero.py.
    if args.action_head:
        action_head_path = args.action_head
    else:
        ckpt = Path(args.checkpoint)
        candidates = [ckpt / "action_head.pt", ckpt.parent / "action_head.pt"]
        action_head_path = next((str(c) for c in candidates if c.exists()), str(candidates[0]))
    print(f"[AWQ] Using action_head: {action_head_path}")

    if args.method == "bnb":
        results = quantize_with_bnb(
            checkpoint_path=args.checkpoint,
            action_head_path=action_head_path,
            output_dir=args.output_dir,
            data_root=args.data_root,
            action_stats=action_stats,
            bits=args.bits,
        )
    else:
        results = quantize_with_awq(
            checkpoint_path=args.checkpoint,
            action_head_path=action_head_path,
            output_dir=args.output_dir,
            data_root=args.data_root,
            action_stats=action_stats,
            bits=args.bits,
            num_calib_samples=args.num_calib_samples,
            group_size=args.group_size,
        )

    summary_path = Path(args.output_dir) / "quantization_report.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[AWQ] Report saved to {summary_path}")
