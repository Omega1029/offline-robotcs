#!/usr/bin/env python3
"""
12_benchmark_latency_memory.py — REAL latency + memory footprint of one policy query, per
quantization mode. Unlike the fake-quant accuracy sweep (which stores bf16 and so cannot
measure size/speed), this loads the model with actually-packed low-bit weights
(bitsandbytes INT8/INT4) to get genuine memory numbers, and also measures the rotation
compute overhead of our scheme.

Modes (one per process for clean peak-memory accounting):
  bf16       — baseline, eval_libero.load_model
  rot_w4a4   — bf16 + Hadamard rotation + fake W4A4 (our scheme's COMPUTE; memory ~bf16
               because fake-quant stores bf16 — reported to quantify rotation overhead)
  bnb_int8   — real 8-bit packed weights (bitsandbytes)
  bnb_int4   — real 4-bit packed weights (bitsandbytes nf4)

Measures, for a single synthetic policy query (~380 tokens + one image, batch 1):
  weights_GB  — GPU memory resident right after load
  peak_GB     — peak GPU memory during forward (weights + activations)
  latency_ms  — median per-query wall time (backbone forward + action-head predict)

CAVEAT: run on an A100-80GB (datacenter), NOT the Jetson Orin Nano deployment target. The
memory FOOTPRINT (weight packing) transfers to edge; absolute LATENCY does not — on a GPU
with fast bf16 tensor cores, INT4 dequant overhead can make low-bit SLOWER, whereas the
speed win is realized on memory-bound edge hardware. bnb-INT4 here is weight-only nf4, a
proxy for our W4 memory (not our exact rotated-W4A4 kernel, which does not yet exist).

Run:  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl ../venv/bin/python 12_benchmark_latency_memory.py --mode bf16
"""
import argparse, json, os, sys, time
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "quant_expirements"))
sys.path.insert(0, str(PIPE / "openvla-oft"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import importlib.util
import numpy as np
import torch
from PIL import Image

_spec = importlib.util.spec_from_file_location("eval_libero", str(PIPE / "02_eval_libero.py"))
eval_libero = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eval_libero)
_spec2 = importlib.util.spec_from_file_location("eval_quant", str(PIPE / "02A_eval_quant_libero.py"))
eval_quant = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(eval_quant)

import quant_advanced as qa

CKPT = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/hf_model")
HEAD = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/action_head.pt")
STATS = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/action_stats.json")
SYS = ("A chat between a curious user and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the user's questions. ")
ACTION_LEN = 56  # OFT_CHUNK_SIZE * OFT_ACTION_DIM


def synthetic_query(processor, model, device):
    """One realistic policy-query input: system+instruction prompt + action tokens + 1 image."""
    instr = "pick up the black bowl and place it on the plate"
    prefix = f"{SYS}USER: What action should the robot take to {instr}? ASSISTANT: "
    pid = processor.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
    iid = torch.cat([pid, model.eval_action_token_ids]).unsqueeze(0).to(device)
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    pv = processor.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, torch.bfloat16)
    return dict(input_ids=iid, attention_mask=torch.ones_like(iid), pixel_values=pv)


@torch.no_grad()
def policy_query(model, inp):
    out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                pixel_values=inp["pixel_values"], output_hidden_states=True)
    h = out.hidden_states[-1][:, -ACTION_LEN:, :]
    return model.action_head.predict_action(h)


def load(mode, device):
    if mode == "bf16":
        processor, model = eval_libero.load_model(CKPT, HEAD, device)
    elif mode == "rot_w4a4":
        processor, model = eval_libero.load_model(CKPT, HEAD, device)
        qa.Rotation(w_bits=4, a_bits=4, kind="hadamard").apply(model)
    elif mode == "bnb_int8":
        processor, model = eval_quant.load_quantized_model(CKPT, HEAD, device, bits=8)
    elif mode == "bnb_int4":
        processor, model = eval_quant.load_quantized_model(CKPT, HEAD, device, bits=4)
    else:
        raise ValueError(mode)
    model.eval()
    return processor, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["bf16", "rot_w4a4", "bnb_int8", "bnb_int4"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="results/bench_latency_memory.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device); torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    processor, model = load(args.mode, device)
    load_s = time.time() - t0
    torch.cuda.synchronize()
    weights_gb = torch.cuda.memory_allocated(device) / 1e9

    inp = synthetic_query(processor, model, device)
    for _ in range(args.warmup):
        policy_query(model, inp)
    torch.cuda.synchronize()

    times = []
    for _ in range(args.iters):
        torch.cuda.synchronize(); s = time.perf_counter()
        policy_query(model, inp)
        torch.cuda.synchronize(); times.append((time.perf_counter() - s) * 1e3)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    times = np.array(times)

    rec = {
        "mode": args.mode, "device": torch.cuda.get_device_name(0),
        "weights_GB": round(weights_gb, 2), "peak_GB": round(peak_gb, 2),
        "latency_ms_median": round(float(np.median(times)), 2),
        "latency_ms_mean": round(float(times.mean()), 2),
        "latency_ms_p90": round(float(np.percentile(times, 90)), 2),
        "throughput_qps": round(1000.0 / float(np.median(times)), 1),
        "load_s": round(load_s, 1), "iters": args.iters,
    }
    print(json.dumps(rec))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
