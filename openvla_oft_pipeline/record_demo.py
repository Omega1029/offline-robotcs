#!/usr/bin/env python3
"""
Record a video of OpenVLA-OFT controlling the LIBERO simulation.

Features:
  - Multi-task: runs N tasks in sequence, joined into one video with title cards
  - Multi-line instruction overlay (word-wraps to fit the frame)
  - Wrist-cam inset (bottom-right corner)
  - Task counter, step counter, SUCCESS/FAILED banner
  - Optional quantization (any ComponentQuant config by name)
  - Optional side-by-side bf16 vs quant comparison

Single task (bf16):
    MUJOCO_GL=egl python record_demo.py \
        --checkpoint checkpoints/openvla_oft_libero/epoch_003 \
        --suite libero_spatial --task_indices 2

Three-task demo (default):
    MUJOCO_GL=egl python record_demo.py \
        --checkpoint checkpoints/openvla_oft_libero/epoch_003

Three-task quantized:
    MUJOCO_GL=egl python record_demo.py \
        --checkpoint checkpoints/openvla_oft_libero/epoch_003 \
        --quant combo_v8_b3a8dct_h8 \
        --output demo_3task_quant.mp4

Side-by-side comparison:
    MUJOCO_GL=egl python record_demo.py \
        --checkpoint checkpoints/openvla_oft_libero/epoch_003 \
        --quant combo_v8_b3a8dct_h8 --compare
"""

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_OPENVLA_OFT_DIR = Path(__file__).resolve().parent / "openvla-oft"
if _OPENVLA_OFT_DIR.is_dir() and str(_OPENVLA_OFT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPENVLA_OFT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

OFT_CHUNK_SIZE = 8
OFT_ACTION_DIM = 7
NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
SUITE_TO_BENCHMARK = {"libero_long": "libero_10"}
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_long":    520,
}

VIDEO_FPS    = 15
SCENE_RES    = 512
WRIST_RES    = 128
FONT_SIZE    = 20
MARGIN       = 10
LINE_SPACING = 6
WRAP_CHARS   = 44   # characters per line at 512px / font-size 20
BANNER_HOLD  = 2    # seconds to hold success/fail banner


def _font(size=FONT_SIZE):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _transparent_rect(img_rgb, x0, y0, x1, y1, fill_rgba):
    """Draw a semi-transparent rectangle on an RGB numpy array."""
    overlay = Image.new("RGBA", (img_rgb.shape[1], img_rgb.shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(x0, y0), (x1, y1)], fill=fill_rgba)
    base = Image.fromarray(img_rgb).convert("RGBA")
    composited = Image.alpha_composite(base, overlay)
    return np.array(composited.convert("RGB"))


def annotate_frame(frame_rgb, instruction, step, max_steps,
                   success=None, quant_label="bf16",
                   task_num=None, total_tasks=None,
                   panel_width=None):
    """
    Annotate a 512×512 (or panel_width wide) RGB frame with:
      - Multi-line instruction header (top)
      - Task counter (top-right)
      - Step counter + model label (bottom-left)
      - SUCCESS / FAILED banner (center, shown when success is not None)
    """
    W = panel_width or frame_rgb.shape[1]
    # characters per line scales with panel width vs default 512
    wrap = max(20, int(WRAP_CHARS * W / 512))

    lines = textwrap.wrap(instruction, width=wrap)
    font      = _font(FONT_SIZE)
    small     = _font(14)
    line_h    = FONT_SIZE + LINE_SPACING
    header_h  = len(lines) * line_h + 2 * MARGIN

    frame_rgb = _transparent_rect(frame_rgb, 0, 0, W, header_h, (0, 0, 0, 190))

    img  = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    # ── instruction lines ─────────────────────────────────────────────────────
    y = MARGIN
    for line in lines:
        draw.text((MARGIN, y), line, fill=(255, 255, 255), font=font)
        y += line_h

    # ── task counter (top-right) ──────────────────────────────────────────────
    if task_num is not None and total_tasks is not None:
        tag = f"Task {task_num}/{total_tasks}"
        bbox = draw.textbbox((0, 0), tag, font=small)
        tw = bbox[2] - bbox[0]
        draw.text((W - tw - MARGIN, MARGIN), tag, fill=(180, 220, 255), font=small)

    # ── bottom bar: step + label ──────────────────────────────────────────────
    active = max(0, step - NUM_STEPS_WAIT)
    bar = f"step {active}/{max_steps}   {quant_label}"
    bar_h = FONT_SIZE + 2 * MARGIN
    img_h = img.height
    frame_rgb = np.array(img)
    frame_rgb = _transparent_rect(frame_rgb, 0, img_h - bar_h, W, img_h, (0, 0, 0, 160))
    img  = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    draw.text((MARGIN, img_h - bar_h + MARGIN // 2), bar,
              fill=(190, 190, 190), font=small)

    # ── center banner ─────────────────────────────────────────────────────────
    if success is True:
        _draw_banner(draw, img, "✓  SUCCESS", (0, 160, 0), font)
    elif success is False:
        _draw_banner(draw, img, "✗  FAILED",  (160, 0, 0), font)

    return np.array(img)


def _draw_banner(draw, img, text, color, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 14
    bw  = tw + 2 * pad
    bh  = th + 2 * pad
    bx  = img.width  // 2 - bw // 2
    by  = img.height // 2 - bh // 2
    draw.rectangle([(bx, by), (bx + bw, by + bh)], fill=color)
    draw.text((bx + pad, by + pad), text, fill=(255, 255, 255), font=font)


def overlay_wrist(scene_rgb, wrist_rgb):
    """Paste a down-sampled wrist-cam in the bottom-right corner."""
    scene = scene_rgb.copy()
    small = np.array(Image.fromarray(wrist_rgb).resize((WRIST_RES, WRIST_RES),
                                                        Image.BILINEAR))
    h, w = scene.shape[:2]
    y0 = h - WRIST_RES - MARGIN
    x0 = w - WRIST_RES - MARGIN
    # white border
    border = 2
    scene[y0 - border : y0 + WRIST_RES + border,
          x0 - border : x0 + WRIST_RES + border] = 255
    scene[y0 : y0 + WRIST_RES, x0 : x0 + WRIST_RES] = small
    return scene


def make_title_card(instruction, task_num, total_tasks,
                    width=SCENE_RES, height=SCENE_RES, duration_sec=1.5):
    """Dark title-card frames shown between tasks."""
    img  = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    font_tag  = _font(18)
    font_main = _font(22)

    # task tag
    tag = f"— Task {task_num} of {total_tasks} —"
    bbox = draw.textbbox((0, 0), tag, font=font_tag)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, height // 3 - 20),
              tag, fill=(140, 180, 240), font=font_tag)

    # instruction wrapped
    wrap = max(20, int(WRAP_CHARS * width / 512))
    lines = textwrap.wrap(instruction, width=wrap)
    line_h = 22 + LINE_SPACING
    total_h = len(lines) * line_h
    y = height // 2 - total_h // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_main)
        tw = bbox[2] - bbox[0]
        draw.text(((width - tw) // 2, y), line, fill=(255, 255, 255), font=font_main)
        y += line_h

    arr = np.array(img)
    n   = int(duration_sec * VIDEO_FPS)
    return [arr.copy() for _ in range(n)]


def make_summary_card(results, width=SCENE_RES, height=SCENE_RES, duration_sec=3.0):
    """Final summary card showing all tasks and outcomes."""
    img  = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    title_font = _font(22)
    row_font   = _font(16)
    small_font = _font(13)

    title = "OpenVLA-OFT  ·  LIBERO Demo"
    bbox  = draw.textbbox((0, 0), title, font=title_font)
    tw    = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, MARGIN * 2), title,
              fill=(200, 220, 255), font=title_font)

    y = height // 4
    for i, (instr, ok, label) in enumerate(results):
        color  = (80, 200, 80) if ok else (200, 80, 80)
        symbol = "✓" if ok else "✗"
        row    = f"{symbol}  Task {i+1}: {textwrap.shorten(instr, 48)}"
        draw.text((MARGIN * 2, y), row, fill=color, font=row_font)
        y += 28

    # model label
    if results:
        _, _, label = results[0]
        note = f"Model: {label}"
        bbox = draw.textbbox((0, 0), note, font=small_font)
        tw   = bbox[2] - bbox[0]
        draw.text(((width - tw) // 2, height - MARGIN * 4),
                  note, fill=(120, 120, 120), font=small_font)

    arr = np.array(img)
    n   = int(duration_sec * VIDEO_FPS)
    return [arr.copy() for _ in range(n)]


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    from prismatic.models.action_heads import L1RegressionActionHead
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from prismatic.vla.action_tokenizer import ActionTokenizer

    ckpt   = Path(checkpoint_path)
    hf_dir = ckpt / "hf_model" if (ckpt / "hf_model").is_dir() else ckpt
    ah_path = ckpt / "action_head.pt"
    if not ah_path.exists():
        raise FileNotFoundError(f"action_head.pt not found at {ah_path}")

    print(f"Loading model from {hf_dir} …")
    processor = AutoProcessor.from_pretrained(str(hf_dir), trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(hf_dir), trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    llm_dim = model.language_model.config.hidden_size
    action_head = L1RegressionActionHead(
        input_dim=llm_dim, hidden_dim=llm_dim, action_dim=OFT_ACTION_DIM,
    ).to(device, dtype=torch.bfloat16)
    action_head.load_state_dict(torch.load(str(ah_path), map_location=device))
    action_head.eval()
    model.action_head = action_head

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    zero_chunk = np.zeros((OFT_CHUNK_SIZE, OFT_ACTION_DIM), dtype=np.float32)
    disc = np.digitize(np.clip(zero_chunk, -1.0, 1.0), action_tokenizer.bins)
    model.eval_action_token_ids = torch.from_numpy(
        (processor.tokenizer.vocab_size - disc).reshape(-1).astype(np.int64)
    )
    return processor, model


def apply_quant(model, quant_name):
    qe_dir = str(Path(__file__).parent / "quant_expirements")
    if qe_dir not in sys.path:
        sys.path.insert(0, qe_dir)
    import quant as q
    import quant_advanced as qa

    def dct(w, a):
        return qa.Transform(kind="dct", w_bits=w, a_bits=a)

    configs = {
        "combo_v8_b3a8dct_h8":   q.ComponentQuant(vision=8, head=8, backbone_tech=dct(3, 8)),
        "combo_v8_b3a4dct_h8":   q.ComponentQuant(vision=8, head=8, backbone_tech=dct(3, 4)),
        "combo_v8_b4dct_h8":     q.ComponentQuant(vision=8, head=8, backbone_tech=dct(4, 4)),
        "combo_vfp8_b3a8dct_h8": q.ComponentQuant(vision="fp8", head=8, backbone_tech=dct(3, 8)),
        "combo_full_int8":        q.ComponentQuant(vision=8, backbone=8, head=8),
        "backbone_w4a4_dct":      q.ComponentQuant(backbone_tech=dct(4, 4)),
    }
    if quant_name not in configs:
        raise ValueError(f"Unknown config '{quant_name}'. Options: {list(configs)}")
    print(f"Applying quantization: {quant_name}")
    configs[quant_name].apply(model)


def load_action_stats(checkpoint_path):
    ckpt = Path(checkpoint_path)
    for p in [ckpt / "action_stats.json", ckpt / "hf_model" / "action_stats.json"]:
        if p.exists():
            with open(p) as f:
                s = json.load(f)
            return (torch.tensor(s["mean"], dtype=torch.float32),
                    torch.tensor(s["std"],  dtype=torch.float32))
    raise FileNotFoundError(f"action_stats.json not found under {checkpoint_path}")


# ── rollout with per-step frame capture ───────────────────────────────────────

def record_rollout(env, model, processor, instruction,
                   action_mean, action_std, init_state, device,
                   seed, max_steps, quant_label="bf16",
                   task_num=None, total_tasks=None):
    """Run one rollout and return (frames_list, success)."""
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)

    sys_msg = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )
    prefix = f"{sys_msg}USER: What action should the robot take to {instruction}? ASSISTANT: "
    prefix_ids = processor.tokenizer(
        prefix, add_special_tokens=True, return_tensors="pt"
    )["input_ids"].squeeze(0)
    input_ids = torch.cat([prefix_ids, model.eval_action_token_ids]).unsqueeze(0).to(device)
    attn_mask = torch.ones_like(input_ids)
    action_len = OFT_CHUNK_SIZE * OFT_ACTION_DIM

    action_mean = action_mean.float().cpu()
    action_std  = action_std.float().cpu()

    def grab(obs, step, success_flag=None):
        scene = obs["agentview_image"][::-1, ::-1].copy()
        wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
        scene = overlay_wrist(scene, wrist)
        return annotate_frame(scene, instruction, step, max_steps,
                              success=success_flag, quant_label=quant_label,
                              task_num=task_num, total_tasks=total_tasks)

    frames  = [grab(obs, 0)]
    done    = False
    success = False
    step    = 0
    cap     = max_steps + NUM_STEPS_WAIT

    while not done and step < cap:
        # settle phase
        if step < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            step += 1
            if done:
                success = True
            frames.append(grab(obs, step))
            continue

        # policy inference
        image = Image.fromarray(obs["agentview_image"][::-1, ::-1])
        pv    = processor.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            out    = model(input_ids=input_ids, attention_mask=attn_mask,
                          pixel_values=pv, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, -action_len:, :]
            chunk  = model.action_head.predict_action(hidden)

        chunk = (torch.as_tensor(np.asarray(chunk.float().cpu()), dtype=torch.float32)
                 .squeeze(0))
        chunk = chunk * action_std + action_mean

        for ac in range(OFT_CHUNK_SIZE):
            if step >= cap or done:
                break
            obs, _, done, _ = env.step(chunk[ac].numpy())
            step += 1
            if done:
                success = True
            frames.append(grab(obs, step))

    # banner hold
    banner = grab(obs, step, success_flag=success)
    frames += [banner.copy() for _ in range(BANNER_HOLD * VIDEO_FPS)]
    return frames, success


# ── video writing ─────────────────────────────────────────────────────────────

def write_video(frames, path, fps=VIDEO_FPS):
    print(f"Writing {len(frames)} frames → {path}  ({len(frames)/fps:.1f}s) …")
    imageio.mimwrite(path, frames, fps=fps, codec="libx264",
                     quality=8, macro_block_size=1,
                     output_params=["-pix_fmt", "yuv420p"])
    size_mb = Path(path).stat().st_size / 1e6
    print(f"Done!  {path}  ({size_mb:.1f} MB)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/openvla_oft_libero/epoch_003")
    ap.add_argument("--suite", default="libero_spatial",
                    choices=["libero_spatial", "libero_object",
                             "libero_goal", "libero_long"])
    ap.add_argument("--task_indices", type=int, nargs="+", default=[0, 4, 7],
                    help="Task indices to record (default: 0 4 7 — visually varied tasks)")
    ap.add_argument("--rollout_indices", type=int, nargs="+", default=None,
                    help="Rollout (init-state) index per task. Repeats last if shorter.")
    ap.add_argument("--quant", default=None,
                    help="Quantization config (default: bf16)")
    ap.add_argument("--compare", action="store_true",
                    help="Side-by-side bf16 vs --quant (requires --quant, single task only)")
    ap.add_argument("--output", default=None,
                    help="Output .mp4 path (auto-named if omitted)")
    ap.add_argument("--fps", type=int, default=VIDEO_FPS)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.compare and args.quant is None:
        ap.error("--compare requires --quant")
    if args.compare and len(args.task_indices) != 1:
        print("Note: --compare only uses the first task index.")
        args.task_indices = [args.task_indices[0]]

    device = torch.device(args.device)

    # resolve rollout indices
    ri_list = args.rollout_indices or [0] * len(args.task_indices)
    while len(ri_list) < len(args.task_indices):
        ri_list.append(ri_list[-1])

    # ── load env resources ────────────────────────────────────────────────────
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench_key = SUITE_TO_BENCHMARK.get(args.suite, args.suite)
    bench     = benchmark.get_benchmark_dict()[bench_key]()
    bddl_base = get_libero_path("bddl_files")
    max_steps = TASK_MAX_STEPS[args.suite]
    action_mean, action_std = load_action_stats(args.checkpoint)

    # resolve output name
    if args.output is None:
        ql = args.quant.replace("_", "") if args.quant else "bf16"
        if args.compare:
            args.output = f"demo_compare_{ql}.mp4"
        elif len(args.task_indices) == 1:
            args.output = f"demo_{args.suite}_t{args.task_indices[0]}_{ql}.mp4"
        else:
            tstr = "_".join(str(t) for t in args.task_indices)
            args.output = f"demo_{args.suite}_tasks{tstr}_{ql}.mp4"

    quant_label = args.quant if args.quant else "bf16"

    # ── load model ────────────────────────────────────────────────────────────
    processor, model = load_model(args.checkpoint, device)

    # ══ SIDE-BY-SIDE MODE ════════════════════════════════════════════════════
    if args.compare:
        task_idx  = args.task_indices[0]
        rollout_i = ri_list[0]
        task      = bench.get_task(task_idx)
        init      = bench.get_task_init_states(task_idx)[rollout_i % 10]
        bddl_file = os.path.join(bddl_base, task.problem_folder, task.bddl_file)
        seed      = rollout_i * 100 + task_idx

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file, camera_heights=SCENE_RES, camera_widths=SCENE_RES
        )
        print(f"\nTask: {task.language}")

        print("[1/2] bf16 rollout …")
        frames_bf16, ok_bf16 = record_rollout(
            env, model, processor, task.language,
            action_mean, action_std, init, device,
            seed=seed, max_steps=max_steps, quant_label="bf16",
            task_num=1, total_tasks=1,
        )
        print(f"  → {'SUCCESS' if ok_bf16 else 'FAILED'}  ({len(frames_bf16)} frames)")

        apply_quant(model, args.quant)

        print(f"[2/2] {args.quant} rollout …")
        frames_q, ok_q = record_rollout(
            env, model, processor, task.language,
            action_mean, action_std, init, device,
            seed=seed, max_steps=max_steps, quant_label=args.quant,
            task_num=1, total_tasks=1,
        )
        print(f"  → {'SUCCESS' if ok_q else 'FAILED'}  ({len(frames_q)} frames)")
        env.close()

        # pad to equal length
        n = max(len(frames_bf16), len(frames_q))
        frames_bf16 += [frames_bf16[-1]] * (n - len(frames_bf16))
        frames_q    += [frames_q[-1]]    * (n - len(frames_q))

        divider = np.ones((SCENE_RES, 4, 3), dtype=np.uint8) * 60
        combined = [np.concatenate([l, divider, r], axis=1)
                    for l, r in zip(frames_bf16, frames_q)]
        write_video(combined, args.output, args.fps)
        return

    # ══ MULTI-TASK MODE ══════════════════════════════════════════════════════
    if args.quant:
        apply_quant(model, args.quant)

    total_tasks = len(args.task_indices)
    all_frames  = []
    results     = []   # (instruction, success, label) for summary card

    for seq_i, (task_idx, rollout_i) in enumerate(
            zip(args.task_indices, ri_list), start=1):

        task      = bench.get_task(task_idx)
        init_list = bench.get_task_init_states(task_idx)
        init      = init_list[rollout_i % len(init_list)]
        bddl_file = os.path.join(bddl_base, task.problem_folder, task.bddl_file)
        seed      = rollout_i * 100 + task_idx

        print(f"\n[{seq_i}/{total_tasks}] Task {task_idx}: {task.language}")

        # title card
        all_frames += make_title_card(task.language, seq_i, total_tasks)

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file, camera_heights=SCENE_RES, camera_widths=SCENE_RES
        )
        frames, ok = record_rollout(
            env, model, processor, task.language,
            action_mean, action_std, init, device,
            seed=seed, max_steps=max_steps, quant_label=quant_label,
            task_num=seq_i, total_tasks=total_tasks,
        )
        env.close()

        print(f"  → {'SUCCESS' if ok else 'FAILED'}  ({len(frames)} frames, "
              f"{len(frames)/args.fps:.1f}s)")
        all_frames += frames
        results.append((task.language, ok, quant_label))

    # summary card
    all_frames += make_summary_card(results)

    write_video(all_frames, args.output, args.fps)
    print(f"\nSummary: {sum(r[1] for r in results)}/{total_tasks} tasks succeeded")


if __name__ == "__main__":
    main()
