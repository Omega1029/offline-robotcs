"""
Evaluate a QUANTIZED OpenVLA-OFT model on the 4 LIBERO task suites.

This is the quantized-inference twin of 02_eval_libero.py. It evaluates either:
    (a) a pre-quantized model directory produced by 03_quantize_awq.py --method bnb
        (the bitsandbytes quantization_config is embedded in its config.json and
        auto-applies on load), or
    (b) a bf16 training checkpoint quantized on-the-fly via --bits {4,8}.

Everything downstream of loading — rollouts, action-head decoding, success bookkeeping,
CSV/JSON output — is identical to 02_eval_libero.py. Only model loading differs.

WHY A SEPARATE LOADER (the fixes carried over from 03_quantize_awq.py):
    bitsandbytes-quantized models cannot be moved with `.to(device)` after load — bnb
    raises "`.to` is not supported for `4-bit` or `8-bit` bitsandbytes models". They must
    be placed at load time via device_map. On top of that, transformers 4.40.x calls
    accelerate.dispatch_model BEFORE it flags the model as quantized, so accelerate's
    single-device "collapse" path runs that forbidden `model.to(device)` anyway. The
    _patch_dispatch_model_for_bnb() shim below forces accelerate onto its hook-dispatch
    path, which correctly places quantized weights AND non-persistent buffers
    (e.g. rotary_emb.inv_freq) on the GPU. See 03_quantize_awq.py for the full writeup.

USAGE:
    # Eval a model already quantized + saved by 03 (recommended)
    MUJOCO_GL=egl python 02A_eval_quant_libero.py \
        --checkpoint checkpoints/openvla_oft_libero_bnb_int8 \
        --action_stats checkpoints/openvla_oft_libero_bnb_int8/action_stats.json \
        --action_head checkpoints/openvla_oft_libero/action_head.pt \
        --output_csv results/libero_eval_bnb_int8.csv \
        --rollouts_per_task 10

    # Quantize a bf16 checkpoint on-the-fly and eval (no separate 03 step)
    MUJOCO_GL=egl python 02A_eval_quant_libero.py \
        --checkpoint checkpoints/openvla_oft_libero/epoch_003/hf_model \
        --action_stats checkpoints/openvla_oft_libero/epoch_003/action_stats.json \
        --bits 8 \
        --rollouts_per_task 3 --max_tasks_per_suite 2

NOTE ON action_head.pt:
    03_quantize_awq.py saves only the VLM; the action head is reconstructed at runtime
    from action_head.pt (kept at bf16, never quantized). Point --action_head at the
    training-time action_head.pt if it isn't found next to the quantized checkpoint.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# openvla-oft ships `prismatic` as a source tree, not an installed package. Make eval
# self-sufficient so it can be run standalone. Mirrors 02_eval_libero.py.
_OPENVLA_OFT_DIR = Path(__file__).resolve().parent / "openvla-oft"
if _OPENVLA_OFT_DIR.is_dir() and str(_OPENVLA_OFT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPENVLA_OFT_DIR))

# Offline by default: the checkpoint's auto_map references openvla/openvla-7b remote
# code, already cached locally. Force offline to avoid network access on every load.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
# The installed LIBERO names the long-horizon suite "libero_10" in its benchmark dict.
SUITE_TO_BENCHMARK = {"libero_long": "libero_10"}
OFT_CHUNK_SIZE = 8
OFT_ACTION_DIM = 7
MAX_STEPS_PER_ROLLOUT = 600
# Do-nothing action used to let the scene settle before the policy takes over.
NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

# Top-level submodules kept at bf16 during on-the-fly quantization (substring match).
# Verified against prismatic/extern/hf/modeling_prismatic.py: self.vision_backbone,
# self.projector, self.language_model. "lm_head" matches language_model.lm_head.
# Must match 03_quantize_awq.py so on-the-fly eval matches the saved-model eval.
BNB_SKIP_MODULES = ["vision_backbone", "projector", "lm_head"]


def _patch_dispatch_model_for_bnb():
    """Bridge a transformers-4.40 / accelerate-1.8 / bnb version mismatch.

    transformers 4.40.x calls accelerate.dispatch_model (modeling_utils.py:3735)
    BEFORE it sets the model's is_loaded_in_8bit / is_loaded_in_4bit flags
    (set later in postprocess_model, line 3738). accelerate >=1.x reads those flags
    at the START of dispatch_model to decide between attaching device hooks and its
    single-device "collapse" path. Because the flags aren't set yet, accelerate takes
    the collapse path, which runs `model.to(device)` — and bitsandbytes forbids that
    ("`.to` is not supported for `4-bit` or `8-bit` bnb models"). (For 4-bit with
    bnb>=0.43.2 accelerate also deliberately skips force_hooks, so the collapse path
    is taken there regardless.)

    Fix: force accelerate onto its hook-dispatch path, which correctly places both
    quantized weights AND non-persistent buffers (e.g. rotary_emb.inv_freq) on the
    target device. We set is_loaded_in_8bit only for the duration of dispatch, then
    restore the prior state so transformers' postprocess records the truthful flags.
    Idempotent; only affects this process. Kept byte-identical to 03_quantize_awq.py.
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


def _make_bnb_config(bits: int) -> BitsAndBytesConfig:
    """bitsandbytes config for on-the-fly quantization, matching 03_quantize_awq.py."""
    if bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=BNB_SKIP_MODULES,
            llm_int8_threshold=6.0,
        )
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=BNB_SKIP_MODULES,
        )
    raise ValueError(f"bnb supports bits in {{4, 8}}, got {bits}")


def load_quantized_model(
    checkpoint_path: str,
    action_head_path: str,
    device: torch.device,
    bits: int | None = None,
):
    """
    Load a quantized OpenVLA-OFT model for eval.

    If the checkpoint already carries a bitsandbytes quantization_config (saved by
    03_quantize_awq.py), it auto-applies on load and `bits` is ignored. Otherwise,
    pass bits={4,8} to quantize a bf16 checkpoint on-the-fly.

    The action_head is NOT part of the HF checkpoint (save_pretrained only serializes
    the VLM, and on reload the OpenVLAForActionPrediction class drops the action_head.*
    keys as "unexpected"). It is reconstructed here from action_head.pt and kept at
    bf16 — it is never quantized. Mirrors 02_eval_libero.py.
    """
    from prismatic.models.action_heads import L1RegressionActionHead

    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Detect a quantization_config baked into the saved checkpoint.
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    has_saved_quant = getattr(config, "quantization_config", None) is not None

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    device_index = device.index if device.index is not None else 0
    quantized = has_saved_quant or (bits is not None)

    if has_saved_quant:
        print("[QuantEval] Checkpoint carries a bitsandbytes quantization_config "
              "— loading as-is (bits flag ignored).")
        # bnb requires placement at load time; no .to(device) afterwards.
        load_kwargs["device_map"] = {"": device_index}
    elif bits is not None:
        print(f"[QuantEval] Quantizing bf16 checkpoint on-the-fly to INT{bits} (bnb).")
        load_kwargs["quantization_config"] = _make_bnb_config(bits)
        load_kwargs["device_map"] = {"": device_index}
    else:
        print("[QuantEval] No saved quant config and no --bits given — loading bf16 "
              "(this is an UNQUANTIZED eval; use --bits to quantize).")

    if quantized:
        # Force accelerate's hook-dispatch path so the forbidden .to() is never called.
        _patch_dispatch_model_for_bnb()

    model = AutoModelForVision2Seq.from_pretrained(checkpoint_path, **load_kwargs)

    # Only a non-quantized model may (and must) be moved with .to(device); a bnb model
    # is already placed via device_map and .to() would raise.
    if not quantized:
        model = model.to(device)
    model.eval()

    assert model.config.model_type == "openvla"

    # Reconstruct the bf16 action head and attach it. Same dims as training.
    llm_dim = model.language_model.config.hidden_size  # 4096 for LLaMA-2-7B
    action_head = L1RegressionActionHead(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=OFT_ACTION_DIM,
    ).to(device, dtype=torch.bfloat16)

    if not Path(action_head_path).exists():
        raise FileNotFoundError(
            f"action_head weights not found at {action_head_path}. 03_quantize_awq.py "
            f"does not copy action_head.pt into the quantized output — point "
            f"--action_head at the training-time action_head.pt."
        )
    state_dict = torch.load(action_head_path, map_location=device)
    action_head.load_state_dict(state_dict)
    action_head.eval()
    model.action_head = action_head

    # Placeholder action-token IDs consumed by run_rollout. Built DIRECTLY to match
    # 01_train_openvla_oft.py exactly (no ID→text→ID round-trip). Mirrors 02_eval_libero.py.
    from prismatic.vla.action_tokenizer import ActionTokenizer

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    zero_chunk = np.zeros((OFT_CHUNK_SIZE, OFT_ACTION_DIM), dtype=np.float32)
    disc = np.digitize(np.clip(zero_chunk, -1.0, 1.0), action_tokenizer.bins)
    action_token_ids = torch.from_numpy(
        (processor.tokenizer.vocab_size - disc).reshape(-1).astype(np.int64)
    )
    assert action_token_ids.numel() == OFT_CHUNK_SIZE * OFT_ACTION_DIM
    model.eval_action_token_ids = action_token_ids  # consumed by run_rollout

    return processor, model


def run_rollout(
    env,
    model,
    processor,
    instruction: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    init_state,
    device: torch.device,
    seed: int = 0,
) -> dict:
    """
    Run a single rollout. Returns dict with keys: success (bool), steps (int), elapsed_sec (float).

    Identical to 02_eval_libero.py — the quantized model exposes the same forward API.
    """
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)

    sys_msg = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )
    prefix = (
        f"{sys_msg}USER: What action should the robot take to {instruction}? "
        f"ASSISTANT: "
    )
    prefix_ids = processor.tokenizer(
        prefix, add_special_tokens=True, return_tensors="pt"
    )["input_ids"].squeeze(0).to(device)
    input_ids = torch.cat([prefix_ids, model.eval_action_token_ids.to(device)]).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    action_len = OFT_CHUNK_SIZE * OFT_ACTION_DIM  # = 56

    action_mean = action_mean.float().cpu()
    action_std = action_std.float().cpu()

    done = False
    success = False
    step = 0
    t0 = time.time()

    while not done and step < MAX_STEPS_PER_ROLLOUT:
        if step < NUM_STEPS_WAIT:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            step += 1
            if done:
                success = True
            continue

        # agentview image is rotated 180° to match training preprocessing.
        image = Image.fromarray(obs["agentview_image"][::-1, ::-1])
        pixel_values = processor.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
            action_hidden = outputs.hidden_states[-1][:, -action_len:, :]
            action_chunk = model.action_head.predict_action(action_hidden)

        action_chunk = torch.as_tensor(
            np.asarray(action_chunk.float().cpu()), dtype=torch.float32
        ).squeeze(0)
        action_chunk = action_chunk * action_std + action_mean  # → world-space actions

        for ac_step in range(OFT_CHUNK_SIZE):
            if step >= MAX_STEPS_PER_ROLLOUT or done:
                break
            obs, reward, done, info = env.step(action_chunk[ac_step].numpy())
            step += 1
            if done:
                success = True

    return {
        "success": success,
        "steps": step,
        "elapsed_sec": time.time() - t0,
    }


def evaluate_suite(
    suite_name: str,
    model,
    processor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
    rollouts_per_task: int,
    max_tasks: int | None,
) -> list[dict]:
    """Evaluate all tasks in a LIBERO suite. Returns list of per-rollout result dicts."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench_key = SUITE_TO_BENCHMARK.get(suite_name, suite_name)
    suite = benchmark.get_benchmark_dict()[bench_key]()
    bddl_base = get_libero_path("bddl_files")

    task_names = suite.get_task_names()
    if max_tasks is not None:
        task_names = task_names[:max_tasks]

    all_results = []

    for task_idx, task_name in enumerate(tqdm(task_names, desc=f"{suite_name} tasks")):
        task = suite.get_task(task_idx)
        init_states = suite.get_task_init_states(task_idx)
        bddl_file = os.path.join(bddl_base, task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file, camera_heights=256, camera_widths=256
        )

        for rollout_idx in range(rollouts_per_task):
            init_state = init_states[rollout_idx % len(init_states)]
            result = run_rollout(
                env=env,
                model=model,
                processor=processor,
                instruction=task.language,
                action_mean=action_mean,
                action_std=action_std,
                init_state=init_state,
                device=device,
                seed=rollout_idx * 100 + task_idx,
            )
            result.update({
                "suite": suite_name,
                "task": task_name,
                "task_idx": task_idx,
                "rollout_idx": rollout_idx,
            })
            all_results.append(result)

        env.close()

        task_successes = [r["success"] for r in all_results if r["task"] == task_name]
        sr = sum(task_successes) / len(task_successes)
        print(f"  {task_name}: {sr:.1%} ({sum(task_successes)}/{len(task_successes)})")

    return all_results


def main(args):
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[Warning] Running on CPU — quantized bnb eval requires CUDA and will fail.")

    # Default: action_head.pt next to the checkpoint, or one level up (training layout).
    if args.action_head:
        action_head_path = args.action_head
    else:
        ckpt = Path(args.checkpoint)
        candidates = [ckpt / "action_head.pt", ckpt.parent / "action_head.pt"]
        action_head_path = next((str(c) for c in candidates if c.exists()), str(candidates[0]))

    print(f"[QuantEval] Loading checkpoint: {args.checkpoint}")
    print(f"[QuantEval] Loading action_head: {action_head_path}")
    processor, model = load_quantized_model(
        args.checkpoint, action_head_path, device, bits=args.bits
    )

    with open(args.action_stats) as f:
        action_stats = json.load(f)
    action_mean = torch.tensor(action_stats["mean"], device=device, dtype=torch.bfloat16)
    action_std = torch.tensor(action_stats["std"], device=device, dtype=torch.bfloat16)

    suites_to_eval = [args.suite] if args.suite else LIBERO_SUITES
    all_results = []

    for suite_name in suites_to_eval:
        print(f"\n[QuantEval] Suite: {suite_name}")
        results = evaluate_suite(
            suite_name=suite_name,
            model=model,
            processor=processor,
            action_mean=action_mean,
            action_std=action_std,
            device=device,
            rollouts_per_task=args.rollouts_per_task,
            max_tasks=args.max_tasks_per_suite,
        )
        all_results.extend(results)

        suite_sr = sum(r["success"] for r in results) / len(results)
        print(f"[QuantEval] {suite_name} SUCCESS RATE: {suite_sr:.1%}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LIBERO Quantized Evaluation Summary")
    print("=" * 60)
    suite_srs = {}
    for suite_name in suites_to_eval:
        suite_results = [r for r in all_results if r["suite"] == suite_name]
        if not suite_results:
            continue
        sr = sum(r["success"] for r in suite_results) / len(suite_results)
        suite_srs[suite_name] = sr
        print(f"  {suite_name:20s}: {sr:.1%} ({sum(r['success'] for r in suite_results)}/{len(suite_results)})")

    if len(suite_srs) == 4:
        avg = sum(suite_srs.values()) / 4
        print(f"  {'AVERAGE':20s}: {avg:.1%}")
        print(f"\n  OFT paper baseline (fp): 97.1% average")
        print(f"  Your quantized result:   {avg:.1%}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["suite", "task", "task_idx", "rollout_idx",
                                                    "success", "steps", "elapsed_sec"])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[QuantEval] Results saved to {args.output_csv}")

        summary_path = Path(args.output_csv).with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "checkpoint": str(args.checkpoint),
                "bits": args.bits,
                "rollouts_per_task": args.rollouts_per_task,
                "suite_success_rates": suite_srs,
                "average_success_rate": sum(suite_srs.values()) / max(len(suite_srs), 1),
            }, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Quantized model dir (output of 03_quantize_awq.py) OR a bf16 "
                        "hf_model/ checkpoint to quantize on-the-fly with --bits.")
    p.add_argument("--action_stats", required=True,
                   help="Path to action_stats.json (saved alongside the checkpoint)")
    p.add_argument("--action_head", default=None,
                   help="Path to action_head.pt (kept at bf16). Defaults to action_head.pt "
                        "next to (or one level above) --checkpoint. For a 03-quantized dir, "
                        "point this at the training-time action_head.pt.")
    p.add_argument("--bits", type=int, default=None, choices=[4, 8],
                   help="Quantize a bf16 checkpoint on-the-fly to INT4/INT8. Ignored if the "
                        "checkpoint already carries a bitsandbytes quantization_config.")
    p.add_argument("--output_csv", default="results/libero_eval_quant.csv")
    p.add_argument("--suite", default=None,
                   choices=LIBERO_SUITES + [None],
                   help="Evaluate a single suite (default: all 4)")
    p.add_argument("--rollouts_per_task", type=int, default=10,
                   help="Paper protocol uses 10. Use 3 for quick smoke-test.")
    p.add_argument("--max_tasks_per_suite", type=int, default=None,
                   help="Limit number of tasks evaluated per suite (for debugging)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
