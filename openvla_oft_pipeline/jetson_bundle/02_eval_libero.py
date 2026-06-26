"""
Evaluate OpenVLA-OFT on all 4 LIBERO task suites following the OFT paper protocol.

Paper protocol (Table 1):
    - 10 rollouts per task (500 per suite × 4 suites = 2000 total rollouts)
    - Action chunk size 8 (same as training)
    - Max 600 environment steps per rollout
    - Success rate = fraction of rollouts where the task success signal fires
    - Evaluate all 50 tasks (10 per suite × 5 suites... wait, it's 4 suites × 10 = 40 tasks)

USAGE:
    # Single GPU eval (recommended — rollouts are not trivially parallelizable in LIBERO)
    MUJOCO_GL=egl python 02_eval_libero.py \
        --checkpoint checkpoints/openvla_oft_libero/epoch_001/hf_model \
        --action_stats checkpoints/openvla_oft_libero/epoch_001/action_stats.json \
        --output_csv results/libero_eval_epoch010.csv \
        --rollouts_per_task 10

    # Quick smoke-test (3 rollouts, 2 tasks per suite):
    MUJOCO_GL=egl python 02_eval_libero.py \
        --checkpoint checkpoints/openvla_oft_libero/epoch_001 \
        --action_stats checkpoints/openvla_oft_libero/epoch_001/action_stats.json \
        --rollouts_per_task 3 --max_tasks_per_suite 2

IMPORTANT NOTES:
    - Sim runs at wall-clock speed; 2000 rollouts × 600 steps ≈ 12-24 hours on a single GPU.
      For paper-speed eval, distribute across multiple workers by setting --suite and running
      4 parallel eval jobs.
    - robosuite version must match training env (1.4.1). Different versions → different
      physics behavior → silent success-rate drops of 5-15%.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# openvla-oft ships `prismatic` as a source tree, not an installed package. Training
# relies on launch_train.sh exporting PYTHONPATH=.../openvla-oft; make eval self-
# sufficient so it can be run standalone.
_OPENVLA_OFT_DIR = Path(__file__).resolve().parent / "openvla-oft"
if _OPENVLA_OFT_DIR.is_dir() and str(_OPENVLA_OFT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPENVLA_OFT_DIR))

# Offline by default: the checkpoint's auto_map references openvla/openvla-7b remote
# code, which Transformers re-checks against the Hub on every load. The code is already
# cached locally, so force offline to avoid any network access (override by exporting
# HF_HUB_OFFLINE=0 if you ever need to refresh the cache).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
# The installed LIBERO names the long-horizon suite "libero_10" in its benchmark dict.
SUITE_TO_BENCHMARK = {"libero_long": "libero_10"}
OFT_CHUNK_SIZE = 8
OFT_ACTION_DIM = 7
MAX_STEPS_PER_ROLLOUT = 600
# Do-nothing action used to let the scene settle before the policy takes over
# (mirrors openvla-oft's run_libero_eval num_steps_wait protocol).
NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def load_model(checkpoint_path: str, action_head_path: str, device: torch.device):
    """
    Load from the HF-format checkpoint saved during training.
    Same class constraints as training — see 01_train_openvla_oft.py for details.

    The OFT action_head is NOT part of the HF checkpoint: save_pretrained() only
    serializes the VLM. Training attaches an L1RegressionActionHead at runtime and
    saves it separately to action_head.pt. We must reconstruct it here, load those
    weights, and reattach it as model.action_head — mirroring 01_train_openvla_oft.py.
    """
    from prismatic.models.action_heads import L1RegressionActionHead

    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # Single GPU eval: use device_map="auto" only if model doesn't fit on one GPU.
        # For OFT-7B in bf16 (~14 GB) on a single 80GB H100 this is fine without it.
    ).to(device)
    model.eval()

    assert model.config.model_type == "openvla"

    # Reconstruct the action head with the same dims used in training and attach it.
    llm_dim = model.language_model.config.hidden_size  # 4096 for LLaMA-2-7B
    action_head = L1RegressionActionHead(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=OFT_ACTION_DIM,
    ).to(device, dtype=torch.bfloat16)

    if not Path(action_head_path).exists():
        raise FileNotFoundError(
            f"action_head weights not found at {action_head_path}. Training saves these "
            f"as action_head.pt next to the epoch checkpoint — pass --action_head."
        )
    state_dict = torch.load(action_head_path, map_location=device)
    action_head.load_state_dict(state_dict)
    action_head.eval()
    model.action_head = action_head

    # Inference mirrors training (01_train_openvla_oft.py): we do NOT use the model's
    # own predict_action (the loaded checkpoint resolves to the original discrete-token
    # openvla-7b remote code, which has no action head). Instead we run a raw forward
    # with output_hidden_states and read the action-token positions. Those positions
    # only exist if the prompt carries ACTION_LEN action tokens, so we append a
    # placeholder (zero-action) chunk — the head reads the hidden states there.
    from prismatic.vla.action_tokenizer import ActionTokenizer

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    # Build placeholder action token IDs DIRECTLY — must match 01_train_openvla_oft.py
    # exactly. The previous ID→text→ID round-trip was not reversible for these special
    # tokens (56 actions came back as 57 tokens) and misaligned the -56: slice.
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

    Inference mirrors training (01_train_openvla_oft.py): the VLM produces hidden
    states at appended action-token positions, the L1RegressionActionHead maps those
    to a normalized action chunk, and we unnormalize with the training mean/std.
    LIBERO success is signalled by the env returning done=True.
    """
    # LIBERO API: reset then apply the fixed initial state; reset() returns obs only,
    # set_init_state() returns the obs to start acting from. step() is a 4-tuple.
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)

    # Build the training-format prompt prefix, then append the placeholder action
    # token IDs directly (see 01_train_openvla_oft.py __getitem__). At inference there
    # is no EOS, so the action tokens are the LAST ACTION_LEN tokens of the sequence.
    # The prefix text is byte-identical to training; the action IDs are appended as
    # raw IDs (never round-tripped through text). The full sequence is constant across
    # steps, so we build it once here.
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
    )["input_ids"].squeeze(0)
    input_ids = torch.cat([prefix_ids, model.eval_action_token_ids]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    action_len = OFT_CHUNK_SIZE * OFT_ACTION_DIM  # = 56

    action_mean = action_mean.float().cpu()
    action_std = action_std.float().cpu()

    done = False
    success = False
    step = 0
    t0 = time.time()

    while not done and step < MAX_STEPS_PER_ROLLOUT:
        # Let objects settle before the policy takes over.
        if step < NUM_STEPS_WAIT:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            step += 1
            if done:
                success = True
            continue

        # agentview image is rotated 180° to match training preprocessing.
        # input_ids / attention_mask are constant across steps (built above).
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
            # Last action_len positions are the action tokens (no EOS at inference).
            action_hidden = outputs.hidden_states[-1][:, -action_len:, :]
            # (1, chunk, action_dim) in mean/std-normalized space
            action_chunk = model.action_head.predict_action(action_hidden)

        action_chunk = torch.as_tensor(
            np.asarray(action_chunk.float().cpu()), dtype=torch.float32
        ).squeeze(0)
        action_chunk = action_chunk * action_std + action_mean  # → world-space actions

        # Execute the full chunk (open-loop temporal smoothing).
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
        print("[Warning] Running on CPU — eval will be extremely slow.")

    # Default: action_head.pt sits next to the checkpoint. If --checkpoint points at
    # the hf_model/ subfolder, the head is one level up (where training wrote it).
    if args.action_head:
        action_head_path = args.action_head
    else:
        ckpt = Path(args.checkpoint)
        candidates = [ckpt / "action_head.pt", ckpt.parent / "action_head.pt"]
        action_head_path = next((str(c) for c in candidates if c.exists()), str(candidates[0]))

    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    print(f"[Eval] Loading action_head: {action_head_path}")
    processor, model = load_model(args.checkpoint, action_head_path, device)

    with open(args.action_stats) as f:
        action_stats = json.load(f)
    action_mean = torch.tensor(action_stats["mean"], device=device, dtype=torch.bfloat16)
    action_std = torch.tensor(action_stats["std"], device=device, dtype=torch.bfloat16)

    suites_to_eval = [args.suite] if args.suite else LIBERO_SUITES
    all_results = []

    for suite_name in suites_to_eval:
        print(f"\n[Eval] Suite: {suite_name}")
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
        print(f"[Eval] {suite_name} SUCCESS RATE: {suite_sr:.1%}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LIBERO Evaluation Summary")
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
        print(f"\n  OFT paper baseline: 97.1% average")
        print(f"  Your result:       {avg:.1%}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["suite", "task", "task_idx", "rollout_idx",
                                                    "success", "steps", "elapsed_sec"])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[Eval] Results saved to {args.output_csv}")

        # Also save a summary JSON for easy paper-table generation
        summary_path = Path(args.output_csv).with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "checkpoint": str(args.checkpoint),
                "rollouts_per_task": args.rollouts_per_task,
                "suite_success_rates": suite_srs,
                "average_success_rate": sum(suite_srs.values()) / max(len(suite_srs), 1),
            }, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to hf_model/ checkpoint directory (saved during training)")
    p.add_argument("--action_stats", required=True,
                   help="Path to action_stats.json saved alongside the checkpoint")
    p.add_argument("--action_head", default=None,
                   help="Path to action_head.pt (saved separately during training). "
                        "Defaults to action_head.pt next to (or one level above) --checkpoint.")
    p.add_argument("--output_csv", default="results/libero_eval.csv")
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
