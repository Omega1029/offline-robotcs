"""
Diagnostic: does the trained L1 action head reproduce ground-truth actions on
TRAINING-distribution images, and does its output vary with the image?

This isolates "training never learned" from "sim/eval mismatch". We feed real
demo frames (exactly as the training dataset would) and compare the head's
prediction to the ground-truth action chunk.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import h5py
from PIL import Image

_OFT = Path(__file__).resolve().parent / "openvla-oft"
sys.path.insert(0, str(_OFT))

from importlib import import_module
eval_mod = import_module("02_eval_libero")

CKPT = "checkpoints/openvla_oft_libero/epoch_003/hf_model"
HEAD = "checkpoints/openvla_oft_libero/epoch_003/action_head.pt"
STATS = "checkpoints/openvla_oft_libero/epoch_003/action_stats.json"
DATA = "/home/justin_williams1/OfflineAutonomy/offline-robotcs/datasets/libero/libero_spatial"
CHUNK, ADIM = 8, 7

device = torch.device("cuda:0")
processor, model = eval_mod.load_model(CKPT, HEAD, device)
stats = json.load(open(STATS))
mean = np.array(stats["mean"]); std = np.array(stats["std"])

# placeholder action token ids (must match training/eval)
from prismatic.vla.action_tokenizer import ActionTokenizer
atok = ActionTokenizer(processor.tokenizer)
zc = np.zeros((CHUNK, ADIM), dtype=np.float32)
disc = np.digitize(np.clip(zc, -1, 1), atok.bins)
action_token_ids = torch.from_numpy((processor.tokenizer.vocab_size - disc).reshape(-1).astype(np.int64))

sys_msg = ("A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions. ")

def predict(image_np, instruction):
    prefix = f"{sys_msg}USER: What action should the robot take to {instruction}? ASSISTANT: "
    prefix_ids = processor.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
    eos = torch.tensor([processor.tokenizer.eos_token_id])
    input_ids = torch.cat([prefix_ids, action_token_ids, eos]).unsqueeze(0).to(device)
    attn = torch.ones_like(input_ids)
    pv = processor.image_processor(images=Image.fromarray(image_np), return_tensors="pt")["pixel_values"].to(device, torch.bfloat16)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, pixel_values=pv, output_hidden_states=True)
        ah = out.hidden_states[-1][:, -(CHUNK*ADIM+1):-1, :]   # training-style slice (EOS at end)
        pred = model.action_head.predict_action(ah)            # (1,8,7) normalized
    return pred.float().cpu().numpy()[0]

f = sorted(Path(DATA).glob("*.hdf5"))[0]
print("file:", f.name)
with h5py.File(f, "r") as h:
    instr = h.attrs.get("problem_info", "")
    instr = instr.decode() if isinstance(instr, bytes) else instr
    demo_keys = list(h["data"].keys())
    d0 = h["data"][demo_keys[0]]
    imgs = d0["obs"]["agentview_rgb"][:]   # (T,H,W,3)
    acts = d0["actions"][:]                # (T,7)
print("instruction:", instr, "| T:", len(acts))

# Compare predicted vs GT (normalized) for 4 timesteps
preds = []
for t in [0, 20, 50, 100]:
    if t >= len(acts): continue
    raw_gt = acts[t:t+CHUNK]
    if len(raw_gt) < CHUNK:
        raw_gt = np.concatenate([raw_gt, np.tile(raw_gt[-1:], (CHUNK-len(raw_gt),1))])
    gt_norm = (raw_gt - mean) / (std + 1e-8)
    p = predict(imgs[t], instr)
    preds.append(p[0])
    print(f"\n-- t={t} --")
    print("pred[0] (norm):", np.round(p[0], 3))
    print("gt[0]   (norm):", np.round(gt_norm[0], 3))
    print("pred[0] world :", np.round(p[0]*std+mean, 3))
    print("gt[0]   world :", np.round(raw_gt[0], 3))
    print("L1(chunk,norm):", round(float(np.abs(p - gt_norm).mean()), 4))

preds = np.array(preds)
print("\n=== variance of pred[0] across the 4 different images (per-dim std) ===")
print(np.round(preds.std(axis=0), 4))
print("If ~0 -> head ignores the image (constant output).")
