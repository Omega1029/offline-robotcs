#!/usr/bin/env python3
"""
On-device (Jetson) OPEN-LOOP eval of the OpenVLA-OFT model against real LIBERO demos.

No simulator required. For sampled timesteps in each demo we feed the demo's own
agentview image + language instruction to the model, predict the 8-step action chunk,
un-normalize it, and compare against the ground-truth actions stored in the demo.
This measures how well the policy reproduces the expert actions (action-prediction
error) — the on-device proxy for the simulator success-rate eval (02_eval_libero.py).

Inference path is byte-for-byte the same as 02_eval_libero.py / 01_train_openvla_oft.py:
  - prompt prefix + appended placeholder action-token IDs
  - raw forward with output_hidden_states; read the last 56 action-token positions
  - L1RegressionActionHead maps those hidden states -> normalized action chunk
  - un-normalize with action_stats mean/std

REQUIREMENTS on the Jetson (inside your venv):
  - torch, transformers, h5py, pillow, numpy
  - openvla-oft installed so `prismatic` is importable  (pip install -e openvla-oft)
  - The model + its matching action head + stats:
        hf_model/            (the exported VLM — you already have this)
        action_head.pt       (NOT inside hf_model — copy it from the SAME epoch
                              checkpoint you exported hf_model from)
        action_stats.json    (you already have this)

USAGE:
  python 05_eval_libero_openloop_jetson.py \
      --checkpoint hf_model \
      --action_head action_head.pt \
      --action_stats action_stats.json \
      --data_root libero_subset \
      --frames_per_demo 16 \
      --output_csv libero_openloop_results.csv
"""
import argparse, json, os, time
from glob import glob
from pathlib import Path

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

    # Placeholder action-token IDs (built exactly as in training/eval — no text round-trip).
    tok = ActionTokenizer(processor.tokenizer)
    zero_chunk = np.zeros((OFT_CHUNK_SIZE, OFT_ACTION_DIM), dtype=np.float32)
    disc = np.digitize(np.clip(zero_chunk, -1.0, 1.0), tok.bins)
    ids = (processor.tokenizer.vocab_size - disc).reshape(-1).astype(np.int64)
    assert ids.size == ACTION_LEN
    model.eval_action_token_ids = torch.from_numpy(ids)
    return processor, model


@torch.no_grad()
def predict_chunk(model, processor, instruction, image_np, action_mean, action_std, device, dtype):
    """Return predicted (chunk, 7) action chunk in world space for one observation."""
    prefix = f"{SYS_MSG}USER: What action should the robot take to {instruction}? ASSISTANT: "
    prefix_ids = processor.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
    input_ids = torch.cat([prefix_ids, model.eval_action_token_ids]).unsqueeze(0).to(device)
    attn = torch.ones_like(input_ids)

    img = Image.fromarray(image_np)  # caller already applied the 180° flip
    pixel_values = processor.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, dtype=dtype)

    out = model(input_ids=input_ids, attention_mask=attn, pixel_values=pixel_values, output_hidden_states=True)
    action_hidden = out.hidden_states[-1][:, -ACTION_LEN:, :]
    chunk = model.action_head.predict_action(action_hidden)          # normalized
    chunk = torch.as_tensor(np.asarray(chunk.float().cpu()), dtype=torch.float32).squeeze(0)
    return chunk * action_std + action_mean                          # world space (chunk, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="hf_model")
    ap.add_argument("--action_head", default="action_head.pt")
    ap.add_argument("--action_stats", default="action_stats.json")
    ap.add_argument("--data_root", default="libero_subset")
    ap.add_argument("--frames_per_demo", type=int, default=16,
                    help="evenly-spaced timesteps sampled per demo (lower = faster)")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--output_csv", default="libero_openloop_results.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    print(f"[Eval] device={device} dtype={args.dtype}")

    stats = json.load(open(args.action_stats))
    action_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    action_std = torch.tensor(stats["std"], dtype=torch.float32)

    processor, model = load_model(args.checkpoint, args.action_head, device, dtype)

    import h5py
    files = sorted(glob(os.path.join(args.data_root, "**", "*.hdf5"), recursive=True))
    if not files:
        raise SystemExit(f"No .hdf5 found under {args.data_root}")
    print(f"[Eval] {len(files)} task files\n")

    rows, all_l1 = [], []
    for f in files:
        with h5py.File(f, "r") as h:
            d = h["data"]
            instruction = json.loads(d.attrs["problem_info"])["language_instruction"]
            instruction = instruction.strip().rstrip(".")
            demo = d[sorted(d.keys(), key=lambda k: int(k.split("_")[1]))[0]]
            gt_actions = demo["actions"][:].astype(np.float32)        # (T, 7)
            images = demo["obs"]["agentview_rgb"]                     # (T, H, W, 3)
            T = gt_actions.shape[0]
            ts = np.linspace(0, T - 1, min(args.frames_per_demo, T)).astype(int)

            errs = []
            t0 = time.time()
            for t in ts:
                img = images[t][::-1, ::-1]                           # same 180° flip as training
                pred = predict_chunk(model, processor, instruction, np.ascontiguousarray(img),
                                     action_mean, action_std, device, dtype).numpy()  # (8,7)
                end = min(t + OFT_CHUNK_SIZE, T)
                gt = gt_actions[t:end]
                if gt.shape[0] < OFT_CHUNK_SIZE:                      # pad like training
                    gt = np.concatenate([gt, np.tile(gt[-1:], (OFT_CHUNK_SIZE - gt.shape[0], 1))])
                errs.append(np.abs(pred - gt))                       # (8,7)
            errs = np.concatenate(errs, axis=0)                      # (n*8, 7)
        task = Path(f).stem.replace("_demo", "")
        l1 = float(errs.mean())
        pos = float(errs[:, 0:3].mean()); rot = float(errs[:, 3:6].mean()); grip = float(errs[:, 6].mean())
        all_l1.append(l1)
        rows.append((task, l1, pos, rot, grip, len(ts), round(time.time() - t0, 1)))
        print(f"  {task[:60]:60s}  L1={l1:.4f}  pos={pos:.4f} rot={rot:.4f} grip={grip:.4f}  ({len(ts)} frames, {rows[-1][-1]}s)")

    overall = float(np.mean(all_l1))
    print(f"\n[Eval] OVERALL mean action L1 (world space) = {overall:.4f}  over {len(files)} tasks")

    with open(args.output_csv, "w") as fo:
        fo.write("task,l1_mean,l1_pos,l1_rot,l1_grip,frames,sec\n")
        for r in rows:
            fo.write(",".join(str(x) for x in r) + "\n")
        fo.write(f"OVERALL,{overall:.6f},,,,,\n")
    print(f"[Eval] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
