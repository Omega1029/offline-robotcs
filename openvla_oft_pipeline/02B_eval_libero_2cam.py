"""
Evaluate the 2-CAMERA + PROPRIO OpenVLA-OFT checkpoints (01B training) on LIBERO.

Mirrors 01B_train_openvla_oft_2cam.py's forward exactly:
    [BOS | patches_agentview(256) | patches_wrist(256) | proprio_token(1) | text...]
with proprio = [ee_pos, quat2axisangle(eef_quat), gripper_qpos] normalized by the
proprio_mean/std stored in action_stats.json.

Also supports --execute_horizon h: execute only the FIRST h actions of each
predicted chunk before re-querying the model (default: full chunk). This is the
inference-side knob for the chunk-size × quantization interaction study — shorter
horizons re-query more often (fresher feedback, more compute), longer horizons
run more open-loop.

USAGE:
    # Full paper-grade eval (400 rollouts, all 4 suites)
    MUJOCO_GL=egl python 02B_eval_libero_2cam.py \
        --checkpoint checkpoints/openvla_oft_libero_2cam/epoch_001/hf_model \
        --action_stats checkpoints/openvla_oft_libero_2cam/epoch_001/action_stats.json \
        --output_csv results/libero_2cam_epoch001.csv

    # Quick screen (spatial only, 3x3)
    MUJOCO_GL=egl python 02B_eval_libero_2cam.py \
        --checkpoint checkpoints/openvla_oft_libero_2cam/epoch_001/hf_model \
        --action_stats checkpoints/openvla_oft_libero_2cam/epoch_001/action_stats.json \
        --suite libero_spatial --rollouts_per_task 3 --max_tasks_per_suite 3
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

_OPENVLA_OFT_DIR = Path(__file__).resolve().parent / "openvla-oft"
if _OPENVLA_OFT_DIR.is_dir() and str(_OPENVLA_OFT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPENVLA_OFT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
SUITE_TO_BENCHMARK = {"libero_long": "libero_10"}
OFT_ACTION_DIM = 7
OFT_PROPRIO_DIM = 8
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_long": 520,
}
NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


# ─────────────────────────────────────────────────────────────────────────────
# 2-cam + proprio multimodal forward — byte-for-byte the same sequence layout as
# 01B_train_openvla_oft_2cam.py::multimodal_forward. Any change there must land
# here too.

def multimodal_forward(vla, input_ids, attention_mask,
                       pixel_values, pixel_values_wrist, proprio):
    input_embeddings = vla.get_input_embeddings()(input_ids)

    patches_agent = vla.projector(vla.vision_backbone(pixel_values))
    patches_wrist = vla.projector(vla.vision_backbone(pixel_values_wrist))
    proprio_tok   = vla.proprio_projector(proprio).unsqueeze(1)

    mm_embeds = torch.cat([
        input_embeddings[:, :1, :],
        patches_agent,
        patches_wrist,
        proprio_tok,
        input_embeddings[:, 1:, :],
    ], dim=1)

    n_extra = patches_agent.shape[1] + patches_wrist.shape[1] + 1
    extra_mask = torch.ones(
        (attention_mask.shape[0], n_extra),
        dtype=attention_mask.dtype, device=attention_mask.device,
    )
    mm_mask = torch.cat([attention_mask[:, :1], extra_mask, attention_mask[:, 1:]], dim=1)

    out = vla.language_model(
        inputs_embeds=mm_embeds,
        attention_mask=mm_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.hidden_states[-1]


def load_model(checkpoint_path: str, ckpt_dir: Path, device: torch.device,
               chunk_size: int):
    """Load HF checkpoint + action_head.pt + proprio_projector.pt (both live in
    the epoch dir, one level above hf_model/)."""
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.vla.action_tokenizer import ActionTokenizer

    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    assert model.config.model_type == "openvla"

    llm_dim = model.language_model.config.hidden_size

    action_head = L1RegressionActionHead(
        input_dim=llm_dim, hidden_dim=llm_dim, action_dim=OFT_ACTION_DIM,
    ).to(device, dtype=torch.bfloat16)
    action_head.load_state_dict(torch.load(ckpt_dir / "action_head.pt", map_location=device))
    action_head.eval()
    model.action_head = action_head

    proprio_projector = ProprioProjector(
        llm_dim=llm_dim, proprio_dim=OFT_PROPRIO_DIM,
    ).to(device, dtype=torch.bfloat16)
    proprio_projector.load_state_dict(
        torch.load(ckpt_dir / "proprio_projector.pt", map_location=device))
    proprio_projector.eval()
    model.proprio_projector = proprio_projector

    # Placeholder action token IDs — identical construction to training.
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    zero_chunk = np.zeros((chunk_size, OFT_ACTION_DIM), dtype=np.float32)
    disc = np.digitize(np.clip(zero_chunk, -1.0, 1.0), action_tokenizer.bins)
    action_token_ids = torch.from_numpy(
        (processor.tokenizer.vocab_size - disc).reshape(-1).astype(np.int64)
    )
    assert action_token_ids.numel() == chunk_size * OFT_ACTION_DIM
    model.eval_action_token_ids = action_token_ids

    return processor, model


def run_rollout(env, model, processor, instruction,
                action_mean, action_std, proprio_mean, proprio_std,
                init_state, device, chunk_size, execute_horizon,
                seed=0, max_steps=520) -> dict:
    import robosuite.utils.transform_utils as T

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
    )["input_ids"].squeeze(0)
    input_ids = torch.cat([prefix_ids, model.eval_action_token_ids]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    action_len = chunk_size * OFT_ACTION_DIM

    action_mean = action_mean.float().cpu()
    action_std = action_std.float().cpu()

    done = False
    success = False
    step = 0
    n_queries = 0
    t0 = time.time()

    cap = max_steps + NUM_STEPS_WAIT
    while not done and step < cap:
        if step < NUM_STEPS_WAIT:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            step += 1
            if done:
                success = True
            continue

        # Both cameras: 180° flip to match training convention.
        image = Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        wrist = Image.fromarray(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
        pixel_values = processor.image_processor(images=image, return_tensors="pt")[
            "pixel_values"].to(device, dtype=torch.bfloat16)
        pixel_values_wrist = processor.image_processor(images=wrist, return_tensors="pt")[
            "pixel_values"].to(device, dtype=torch.bfloat16)

        proprio_np = np.concatenate([
            obs["robot0_eef_pos"],
            T.quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]).astype(np.float32)
        proprio_np = (proprio_np - proprio_mean) / (proprio_std + 1e-8)
        proprio_t = torch.from_numpy(proprio_np).unsqueeze(0).to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            last_hidden = multimodal_forward(
                model, input_ids, attention_mask,
                pixel_values, pixel_values_wrist, proprio_t,
            )
            # No EOS at inference — action tokens are the last action_len positions.
            action_hidden = last_hidden[:, -action_len:, :]
            action_chunk = model.action_head.predict_action(action_hidden)
        n_queries += 1

        action_chunk = action_chunk.float().cpu().squeeze(0)
        action_chunk = action_chunk * action_std + action_mean

        # Execute only the first `execute_horizon` actions, then re-query.
        for ac_step in range(min(execute_horizon, chunk_size)):
            if step >= cap or done:
                break
            obs, reward, done, info = env.step(action_chunk[ac_step].numpy())
            step += 1
            if done:
                success = True

    return {
        "success": success,
        "steps": step,
        "n_queries": n_queries,
        "elapsed_sec": time.time() - t0,
    }


def evaluate_suite(suite_name, model, processor, action_mean, action_std,
                   proprio_mean, proprio_std, device, rollouts_per_task,
                   max_tasks, chunk_size, execute_horizon) -> list:
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
                env=env, model=model, processor=processor,
                instruction=task.language,
                action_mean=action_mean, action_std=action_std,
                proprio_mean=proprio_mean, proprio_std=proprio_std,
                init_state=init_state, device=device,
                chunk_size=chunk_size, execute_horizon=execute_horizon,
                seed=rollout_idx * 100 + task_idx,
                max_steps=TASK_MAX_STEPS.get(suite_name, 520),
            )
            result.update({
                "suite": suite_name, "task": task_name,
                "task_idx": task_idx, "rollout_idx": rollout_idx,
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

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt = Path(args.checkpoint)
    ckpt_dir = ckpt.parent if ckpt.name == "hf_model" else ckpt

    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    processor, model = load_model(args.checkpoint, ckpt_dir, device, args.chunk_size)

    # ── Optional fake-quantization of the backbone (chunk × quant study) ──────
    # Reuses the exact machinery from the mono-cam sweep (quant_expirements/):
    # Transform("dct", w, a) wraps every backbone linear with the DCT rotation +
    # W{w}A{a} fake-quant. Vision tower / projector / heads stay bf16.
    if args.quant != "none":
        _QE_DIR = Path(__file__).resolve().parent / "quant_expirements"
        if str(_QE_DIR) not in sys.path:
            sys.path.insert(0, str(_QE_DIR))
        from quant_advanced import Transform
        w_bits, a_bits = {"w4a4_dct": (4, 4), "w3a8_dct": (3, 8)}[args.quant]
        Transform("dct", w_bits=w_bits, a_bits=a_bits).apply(model)

    with open(args.action_stats) as f:
        action_stats = json.load(f)
    action_mean = torch.tensor(action_stats["mean"], device=device, dtype=torch.bfloat16)
    action_std = torch.tensor(action_stats["std"], device=device, dtype=torch.bfloat16)
    proprio_mean = np.asarray(action_stats["proprio_mean"], dtype=np.float32)
    proprio_std = np.asarray(action_stats["proprio_std"], dtype=np.float32)

    execute_horizon = args.execute_horizon or args.chunk_size
    print(f"[Eval] chunk_size={args.chunk_size}, execute_horizon={execute_horizon}")

    suites_to_eval = [args.suite] if args.suite else LIBERO_SUITES
    all_results = []

    for suite_name in suites_to_eval:
        print(f"\n[Eval] Suite: {suite_name}")
        results = evaluate_suite(
            suite_name=suite_name, model=model, processor=processor,
            action_mean=action_mean, action_std=action_std,
            proprio_mean=proprio_mean, proprio_std=proprio_std,
            device=device, rollouts_per_task=args.rollouts_per_task,
            max_tasks=args.max_tasks_per_suite,
            chunk_size=args.chunk_size, execute_horizon=execute_horizon,
        )
        all_results.extend(results)
        suite_sr = sum(r["success"] for r in results) / len(results)
        print(f"[Eval] {suite_name} SUCCESS RATE: {suite_sr:.1%}")

    print("\n" + "=" * 60)
    print("LIBERO 2-cam Evaluation Summary")
    print("=" * 60)
    suite_srs = {}
    for suite_name in suites_to_eval:
        suite_results = [r for r in all_results if r["suite"] == suite_name]
        if not suite_results:
            continue
        sr = sum(r["success"] for r in suite_results) / len(suite_results)
        suite_srs[suite_name] = sr
        print(f"  {suite_name:20s}: {sr:.1%} "
              f"({sum(r['success'] for r in suite_results)}/{len(suite_results)})")

    if len(suite_srs) == 4:
        avg = sum(suite_srs.values()) / 4
        print(f"  {'AVERAGE':20s}: {avg:.1%}")
        print("\n  Per-suite bars (near-SOTA): spatial/object/goal >95%, long 90-95%")
        print(f"  OFT paper (2-cam+proprio): 97.1% average")

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "suite", "task", "task_idx", "rollout_idx",
                "success", "steps", "n_queries", "elapsed_sec"])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[Eval] Results saved to {args.output_csv}")

        summary_path = Path(args.output_csv).with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "checkpoint": str(args.checkpoint),
                "chunk_size": args.chunk_size,
                "execute_horizon": execute_horizon,
                "quant": args.quant,
                "rollouts_per_task": args.rollouts_per_task,
                "suite_success_rates": suite_srs,
                "average_success_rate": sum(suite_srs.values()) / max(len(suite_srs), 1),
            }, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to hf_model/ dir of a 01B (2-cam) epoch checkpoint")
    p.add_argument("--action_stats", required=True)
    p.add_argument("--output_csv", default="results/libero_2cam_eval.csv")
    p.add_argument("--suite", default=None, choices=LIBERO_SUITES + [None])
    p.add_argument("--rollouts_per_task", type=int, default=10)
    p.add_argument("--max_tasks_per_suite", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=8,
                   help="Must match the checkpoint's training chunk size")
    p.add_argument("--execute_horizon", type=int, default=None,
                   help="Execute only the first h actions per chunk before re-querying "
                        "(default: full chunk). The knob for the chunk x quant study.")
    p.add_argument("--quant", default="none",
                   choices=["none", "w4a4_dct", "w3a8_dct"],
                   help="Fake-quantize the backbone with DCT rotation before eval.")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
