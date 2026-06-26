#!/usr/bin/env python3
"""
On-device (Jetson) CLOSED-LOOP eval of the OpenVLA-OFT model in the LIBERO simulator.

Unlike 05_eval_libero_openloop_jetson.py (which only measures action-prediction L1
against expert demos), this rolls the policy out in the actual LIBERO MuJoCo simulator
and reports the field-standard metric: TASK SUCCESS RATE.

Per task we reset to each benchmark initial state, then loop:
  observe agentview image -> predict 8-step action chunk -> execute the chunk in the
  sim -> requery. An episode is a success if the sim reports `done` before max steps.

Inference path is byte-for-byte the same as 05_eval_libero_openloop_jetson.py:
  - single agentview image, 180 deg flip, prompt prefix + 56 placeholder action tokens
  - raw forward with output_hidden_states; L1RegressionActionHead on the last 56 positions
  - un-normalize with action_stats mean/std -> raw demo action space, fed straight to env

The predicted (un-normalized) action is already in the same space as the recorded demo
actions the model was trained on, so it is sent to env.step() directly with NO gripper
remapping (that matches how the demos that produced action_stats were collected).

REQUIREMENTS on the Jetson (inside your venv), in addition to 05's requirements:
  - mujoco==2.3.7, robosuite==1.4.1, bddl, gym, easydict, cloudpickle, imageio[ffmpeg]
  - LIBERO installed so `libero` is importable  (pip install -e --no-deps LIBERO)
  - Headless GPU rendering: MUJOCO_GL=egl (set automatically below)

USAGE:
  MUJOCO_GL=egl python 06_eval_libero_closedloop_jetson.py \
      --checkpoint hf_model \
      --action_head action_head.pt \
      --action_stats action_stats.json \
      --task_suite_name libero_spatial \
      --num_trials_per_task 5 \
      --task_id -1 \
      --output_csv libero_closedloop_results.csv

Tip: start with `--task_id 0 --num_trials_per_task 1` as a smoke test to confirm the
simulator renders on the Jetson before launching a full suite (which is slow on-device).
"""
import argparse, csv, json, os, time
from collections import deque
from pathlib import Path

# Must be set before MuJoCo/robosuite import so offscreen rendering uses the Jetson GPU.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from PIL import Image

OFT_CHUNK_SIZE = 8
OFT_ACTION_DIM = 7
ACTION_LEN = OFT_CHUNK_SIZE * OFT_ACTION_DIM  # 56
SYS_MSG = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)

# Max env steps per suite (mirrors openvla-oft/run_libero_eval.py TASK_MAX_STEPS).
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]  # no-op while objects settle


# --------------------------------------------------------------------------------------
# Model loading + inference (identical to 05_eval_libero_openloop_jetson.py)
# --------------------------------------------------------------------------------------
def load_model(checkpoint, action_head_path, device, dtype):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.vla.action_tokenizer import ActionTokenizer

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    assert model.config.model_type == "openvla", model.config.model_type

    llm_dim = model.language_model.config.hidden_size
    head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=OFT_ACTION_DIM)
    head = head.to(device, dtype=dtype)
    if not Path(action_head_path).exists():
        raise FileNotFoundError(
            f"action_head not found at {action_head_path}. It is saved separately from "
            f"hf_model — copy action_head.pt from the same epoch checkpoint."
        )
    head.load_state_dict(torch.load(action_head_path, map_location=device))
    head.eval()
    model.action_head = head

    tok = ActionTokenizer(processor.tokenizer)
    zero_chunk = np.zeros((OFT_CHUNK_SIZE, OFT_ACTION_DIM), dtype=np.float32)
    disc = np.digitize(np.clip(zero_chunk, -1.0, 1.0), tok.bins)
    ids = (processor.tokenizer.vocab_size - disc).reshape(-1).astype(np.int64)
    assert ids.size == ACTION_LEN
    model.eval_action_token_ids = torch.from_numpy(ids)
    return processor, model


def center_crop(image_np, crop_scale=0.9):
    """Center-crop to `crop_scale` area (matches OFT training random-crop aug, crop_scale=0.9).

    Faithful NumPy/PIL replica of openvla-oft's center_crop_image (sqrt(crop_scale) of H/W,
    centered), avoiding the TensorFlow dependency. The processor resizes afterward.
    """
    H, W = image_np.shape[:2]
    frac = crop_scale ** 0.5
    new_h, new_w = int(round(frac * H)), int(round(frac * W))
    top, left = (H - new_h) // 2, (W - new_w) // 2
    return np.ascontiguousarray(image_np[top:top + new_h, left:left + new_w])


@torch.no_grad()
def predict_chunk(model, processor, instruction, image_np, action_mean, action_std, device, dtype,
                  use_center_crop=False):
    """Return predicted (chunk, 7) action chunk in world space for one observation."""
    prefix = f"{SYS_MSG}USER: What action should the robot take to {instruction}? ASSISTANT: "
    prefix_ids = processor.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
    input_ids = torch.cat([prefix_ids, model.eval_action_token_ids]).unsqueeze(0).to(device)
    attn = torch.ones_like(input_ids)

    if use_center_crop:
        image_np = center_crop(image_np)
    img = Image.fromarray(image_np)  # caller already applied the 180 deg flip
    pixel_values = processor.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, dtype=dtype)

    out = model(input_ids=input_ids, attention_mask=attn, pixel_values=pixel_values, output_hidden_states=True)
    action_hidden = out.hidden_states[-1][:, -ACTION_LEN:, :]
    chunk = model.action_head.predict_action(action_hidden)          # normalized
    chunk = torch.as_tensor(np.asarray(chunk.float().cpu()), dtype=torch.float32).squeeze(0)
    return (chunk * action_std + action_mean).numpy()                # world space (chunk, 7)


# --------------------------------------------------------------------------------------
# LIBERO env helpers (inlined to avoid libero_utils' TensorFlow import)
# --------------------------------------------------------------------------------------
def make_env(task, resolution):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution
    )
    env.seed(0)
    return env, task.language


def get_agentview(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # 180 deg flip = train preproc


def run_episode(env, instruction, initial_state, model, processor,
                action_mean, action_std, device, dtype, max_steps, num_steps_wait,
                use_center_crop=False):
    env.reset()
    obs = env.set_init_state(initial_state)
    queue = deque(maxlen=OFT_CHUNK_SIZE)

    t, success = 0, False
    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:  # let objects settle
                obs, _, done, _ = env.step(DUMMY_ACTION)
                t += 1
                continue

            if len(queue) == 0:
                chunk = predict_chunk(model, processor, instruction, get_agentview(obs),
                                      action_mean, action_std, device, dtype,
                                      use_center_crop=use_center_crop)  # (8,7) world space
                queue.extend(chunk)

            action = np.clip(queue.popleft(), -1.0, 1.0)  # raw demo action space -> straight to env
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
    ap.add_argument("--task_suite_name", default="libero_spatial",
                    choices=list(TASK_MAX_STEPS.keys()))
    ap.add_argument("--task_id", type=int, default=-1,
                    help="single task index to run; -1 runs all tasks in the suite")
    ap.add_argument("--num_trials_per_task", type=int, default=5)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--center_crop", action="store_true",
                    help="apply OFT 0.9 center crop to the policy input (matches training aug)")
    ap.add_argument("--env_img_res", type=int, default=256)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--output_csv", default="libero_closedloop_results.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    print(f"[Eval] device={device} dtype={args.dtype} MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

    stats = json.load(open(args.action_stats))
    action_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    action_std = torch.tensor(stats["std"], dtype=torch.float32)

    processor, model = load_model(args.checkpoint, args.action_head, device, dtype)

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks = suite.n_tasks
    task_ids = range(num_tasks) if args.task_id < 0 else [args.task_id]
    max_steps = TASK_MAX_STEPS[args.task_suite_name]
    print(f"[Eval] suite={args.task_suite_name}  tasks={list(task_ids)}  "
          f"trials/task={args.num_trials_per_task}  max_steps={max_steps}\n")

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
                             action_mean, action_std, device, dtype,
                             max_steps, args.num_steps_wait, use_center_crop=args.center_crop)
            succ += int(ok)
            grand_eps += 1
            grand_succ += int(ok)
            print(f"  task {task_id} ep {ep + 1}/{n_trials}  "
                  f"success={ok}  running={succ}/{ep + 1}")
        env.close()

        rate = succ / n_trials if n_trials else 0.0
        rows.append((task_id, instruction[:60], succ, n_trials, round(rate, 3),
                     round(time.time() - t0, 1)))
        print(f"  >> task {task_id} '{instruction[:50]}'  "
              f"success_rate={rate:.3f} ({succ}/{n_trials})\n")

    overall = grand_succ / grand_eps if grand_eps else 0.0
    print(f"[Eval] OVERALL success rate = {overall:.4f} ({grand_succ}/{grand_eps}) "
          f"on {args.task_suite_name}")

    with open(args.output_csv, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["task_id", "instruction", "successes", "trials", "success_rate", "sec"])
        w.writerows(rows)
        w.writerow(["OVERALL", "", grand_succ, grand_eps, round(overall, 6), ""])
    print(f"[Eval] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
