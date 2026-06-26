#!/usr/bin/env python3
"""
On-device (Jetson) CLOSED-LOOP LIBERO eval using the CORRECT OpenVLA-OFT inference path.

This is the fix for 06_eval_libero_closedloop_jetson.py, which (like 05) hand-rolled the
forward pass and silently used the *original* OpenVLA autoregressive/causal inference
instead of the OFT parallel-decoding path. That produced ~14% success vs ~90% on H100.

The difference is entirely in how actions are predicted. The OFT model
(prismatic.extern.hf.modeling_prismatic.OpenVLAForActionPrediction.predict_action) does:
  - zero out the action-token embeddings before the forward pass
  - append the special empty token (29871) and a STOP token, as seen at train time
  - run BIDIRECTIONAL self-attention over the action block (not causal)
  - read action hidden states at the prompt offset (not the last 56 positions)
The hand-rolled path in 05/06 did none of these. Here we load the model as the OFT class
(exactly as openvla-oft's get_vla does) and call its real predict_action.

Normalization: this fine-tune uses mean/std stats from action_stats.json (the model's
baked-in norm_stats has no 'libero' key). We inject identity bounds (q01=-1, q99=+1) so
predict_action's internal un-normalization is a no-op, then apply mean/std ourselves --
identical to 05's un-normalization. The ONLY thing changed vs 06 is the (now correct)
forward pass.

USAGE:
  MUJOCO_GL=egl python 07_eval_libero_closedloop_oft_jetson.py \
      --checkpoint hf_model --action_head action_head.pt --action_stats action_stats.json \
      --task_suite_name libero_spatial --task_id -1 --num_trials_per_task 50 \
      --output_csv libero_closedloop_oft_results.csv
"""
import argparse, csv, json, os, time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from PIL import Image

OFT_CHUNK_SIZE = 8
OFT_ACTION_DIM = 7
SYS_MSG = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)

TASK_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
UNNORM_KEY = "libero"  # identity-stats key we inject (see module docstring)


# --------------------------------------------------------------------------------------
# Model loading (mirrors openvla-oft get_vla: register OFT classes, then from_pretrained)
# --------------------------------------------------------------------------------------
def load_oft_model(checkpoint, action_head_path, device, dtype):
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.action_heads import L1RegressionActionHead

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.vision_backbone.set_num_images_in_input(1)
    assert type(model).__name__ == "OpenVLAForActionPrediction"
    assert hasattr(model, "_prepare_input_for_action_prediction"), "not the OFT modeling!"

    llm_dim = model.language_model.config.hidden_size
    head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=OFT_ACTION_DIM)
    head = head.to(device, dtype=dtype)
    if not Path(action_head_path).exists():
        raise FileNotFoundError(f"action_head not found at {action_head_path}")
    head.load_state_dict(torch.load(action_head_path, map_location=device))
    head.eval()
    model.action_head = head

    # Inject identity bounds so predict_action's q01/q99 un-normalization is a no-op;
    # we apply the fine-tune's mean/std afterward (matches 05's un-normalization).
    ones = [1.0] * OFT_ACTION_DIM
    model.norm_stats = {UNNORM_KEY: {"action": {"q01": [-1.0] * OFT_ACTION_DIM, "q99": ones,
                                                "mask": [True] * OFT_ACTION_DIM}}}
    return processor, model


@torch.no_grad()
def predict_chunk(model, processor, instruction, image_np, action_mean, action_std, device, dtype,
                  center_crop=True):
    """Return predicted (chunk, 7) action chunk in world space via the OFT predict_action path."""
    prefix = f"{SYS_MSG}USER: What action should the robot take to {instruction}? ASSISTANT: "
    input_ids = processor.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)

    img = Image.fromarray(image_np)  # caller already applied the 180 deg flip
    if center_crop:
        # OFT eval-time center crop (central sqrt(0.9) area), matching
        # openvla_utils.center_crop_image; models trained w/ random-crop aug need this.
        s = 0.9 ** 0.5
        w, h = img.size
        img = img.crop((round((1 - s) / 2 * w), round((1 - s) / 2 * h),
                        round((1 + s) / 2 * w), round((1 + s) / 2 * h)))
    pixel_values = processor.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, dtype=dtype)

    # predict_action appends the action placeholder tokens + STOP, zeros action embeds,
    # runs bidirectional attention, and reads the action hidden states at the right offset.
    norm_chunk, _ = model.predict_action(
        input_ids=input_ids,
        unnorm_key=UNNORM_KEY,                 # identity bounds -> returns normalized chunk
        action_head=model.action_head,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
    )
    chunk = torch.as_tensor(np.asarray(norm_chunk), dtype=torch.float32)  # (8,7) normalized
    return (chunk * action_std + action_mean).numpy()                    # world space (8,7)


# --------------------------------------------------------------------------------------
# LIBERO env helpers + rollout (identical harness to 06)
# --------------------------------------------------------------------------------------
def make_env(task, resolution):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    return env, task.language


def get_agentview(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # 180 deg flip = train preproc


def run_episode(env, instruction, initial_state, model, processor,
                action_mean, action_std, device, dtype, max_steps, num_steps_wait,
                center_crop=True):
    env.reset()
    obs = env.set_init_state(initial_state)
    queue = deque(maxlen=OFT_CHUNK_SIZE)
    t, success = 0, False
    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(DUMMY_ACTION)
                t += 1
                continue
            if len(queue) == 0:
                chunk = predict_chunk(model, processor, instruction, get_agentview(obs),
                                      action_mean, action_std, device, dtype, center_crop)
                queue.extend(chunk)
            action = np.clip(queue.popleft(), -1.0, 1.0)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        print(f"    [episode error] {e}")
    return success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="hf_model")
    ap.add_argument("--action_head", default="action_head.pt")
    ap.add_argument("--action_stats", default="action_stats.json")
    ap.add_argument("--task_suite_name", default="libero_spatial", choices=list(TASK_MAX_STEPS.keys()))
    ap.add_argument("--task_id", type=int, default=-1)
    ap.add_argument("--max_tasks", type=int, default=0,
                    help="When --task_id<0, cap the number of tasks evaluated (0=all). For quick runs.")
    ap.add_argument("--num_trials_per_task", type=int, default=50)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--center_crop", type=int, default=1,
                    help="1=apply OFT eval-time center crop (matches standard run_libero_eval; "
                         "required for models trained with random-crop aug). 0=off (legacy 07_ behavior).")
    ap.add_argument("--env_img_res", type=int, default=256)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--output_csv", default="libero_closedloop_oft_results.csv")
    ap.add_argument("--quant_bits", type=int, default=0,
                    help="If >0 and <16, fake-quantize the language_model backbone linears to "
                         "this many bits (weight-only, per-output-channel symmetric) before eval. "
                         "0=off (bf16 reference). Use 4 for the INT4 accuracy number.")
    ap.add_argument("--rotation", type=str, default="none",
                    choices=["none", "dct", "hadamard", "random_orthogonal"],
                    help="If set (with --quant_bits>0), apply QuaRot/SpinQuant-style incoherence "
                         "rotation before weight quant (RotatedQuantLinear). 'dct' = orthonormal "
                         "DCT-II, valid for all dims. 'none' = plain per-channel RTN.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    print(f"[Eval] OFT inference | device={device} dtype={args.dtype} MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

    stats = json.load(open(args.action_stats))
    action_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    action_std = torch.tensor(stats["std"], dtype=torch.float32)

    processor, model = load_oft_model(args.checkpoint, args.action_head, device, dtype)

    if 0 < args.quant_bits < 16:
        import sys as _sys
        _qp = str(Path(__file__).resolve().parent / "quant_bench" / "quant_expirements")
        if _qp not in _sys.path:
            _sys.path.insert(0, _qp)
        from quant import backbone_linears
        n_lin = len(list(backbone_linears(model)))
        if args.rotation != "none":
            from quant_advanced import Rotation
            Rotation(w_bits=args.quant_bits, a_bits=None, kind=args.rotation).apply(model)
            print(f"[Eval] ROTATED-QUANT W{args.quant_bits} + {args.rotation.upper()} rotation "
                  f"(QuaRot/SpinQuant-style) applied to {n_lin} language_model linears")
        else:
            from quant import FakeQuant
            FakeQuant(w_bits=args.quant_bits, a_bits=None).apply(model)
            print(f"[Eval] FAKE-QUANT W{args.quant_bits} (weight-only, per-out-channel symmetric) "
                  f"applied to {n_lin} language_model linears — INT4 accuracy proxy for GGUF Q4")

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks = suite.n_tasks
    task_ids = range(num_tasks) if args.task_id < 0 else [args.task_id]
    if args.task_id < 0 and args.max_tasks > 0:
        task_ids = list(task_ids)[:args.max_tasks]
    max_steps = TASK_MAX_STEPS[args.task_suite_name]
    print(f"[Eval] suite={args.task_suite_name} tasks={list(task_ids)} "
          f"trials/task={args.num_trials_per_task} max_steps={max_steps}\n")

    rows, grand_eps, grand_succ = [], 0, 0
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, instruction = make_env(task, args.env_img_res)
        instruction = instruction.strip().rstrip(".")
        n_trials = min(args.num_trials_per_task, len(init_states))
        succ, t0 = 0, time.time()
        for ep in range(n_trials):
            ok = run_episode(env, instruction, init_states[ep], model, processor,
                             action_mean, action_std, device, dtype, max_steps, args.num_steps_wait,
                             bool(args.center_crop))
            succ += int(ok); grand_eps += 1; grand_succ += int(ok)
            print(f"  task {task_id} ep {ep + 1}/{n_trials}  success={ok}  running={succ}/{ep + 1}")
        env.close()
        rate = succ / n_trials if n_trials else 0.0
        rows.append((task_id, instruction[:60], succ, n_trials, round(rate, 3), round(time.time() - t0, 1)))
        print(f"  >> task {task_id} '{instruction[:50]}'  success_rate={rate:.3f} ({succ}/{n_trials})\n")

    overall = grand_succ / grand_eps if grand_eps else 0.0
    print(f"[Eval] OVERALL success rate = {overall:.4f} ({grand_succ}/{grand_eps}) on {args.task_suite_name}")

    with open(args.output_csv, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["task_id", "instruction", "successes", "trials", "success_rate", "sec"])
        w.writerows(rows)
        w.writerow(["OVERALL", "", grand_succ, grand_eps, round(overall, 6), ""])
    print(f"[Eval] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
