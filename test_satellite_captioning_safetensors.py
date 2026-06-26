"""
Benchmark / smoke-test the satellite captioning model (SmolVLM + LoRA safetensors)
from fine_tune_satellite_vlm_captioning.py — same DATA_ROOT, instruction, and chat
layout as training. No llama.cpp / GGUF.

Default adapter dir: ./smolvlm_satellite_captioning (adapter_model.safetensors).
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import os
import random
import time
import tokenize

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

# Keep in sync with fine_tune_satellite_vlm_captioning.py
MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
OUTPUT_DIR = "./smolvlm_satellite_captioning"
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "satellite"))
FIXED_INSTRUCTION = "Describe this satellite image in detail."
SEED = 42


def parse_caption_strings(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw.startswith("["):
        return []
    out: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(raw).readline):
            if tok.type == tokenize.STRING:
                out.append(ast.literal_eval(tok.string))
    except (tokenize.TokenError, SyntaxError, ValueError):
        return []
    return out


def load_split(csv_path: str, data_root: str):
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row["filepath"].strip()
            captions = parse_caption_strings(row["captions"])
            if not captions:
                continue
            img_path = os.path.join(data_root, fp)
            if not os.path.isfile(img_path):
                continue
            records.append(
                {
                    "image": os.path.abspath(img_path),
                    "captions": captions,
                }
            )
    return records


def build_messages(instruction: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Instruction: {instruction}"},
            ],
        },
    ]


class SatelliteCaptionModel:
    def __init__(
        self,
        *,
        base_model_id: str,
        adapter_dir: str | None,
        merge_lora: bool,
        device: str,
        dtype: torch.dtype,
    ):
        proc_src = adapter_dir if adapter_dir and os.path.isdir(adapter_dir) else base_model_id
        print(f"Loading processor from: {proc_src}")
        self.processor = AutoProcessor.from_pretrained(proc_src)

        print(f"Loading base model: {base_model_id} ({device}, {dtype})")
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

        if adapter_dir and os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
            print(f"Loading LoRA adapter from: {adapter_dir}")
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            if merge_lora:
                print("Merging LoRA into base weights...")
                self.model = self.model.merge_and_unload()
        elif adapter_dir:
            print(
                f"No adapter_config.json under {adapter_dir!r} — using base weights only."
            )

        self.model.eval()
        self.device = device

    def caption(self, image_path: str, instruction: str, max_new_tokens: int) -> str:
        img = Image.open(image_path).convert("RGB")
        messages = build_messages(instruction)
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=[prompt], images=[img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = out_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Satellite VLM captioning (safetensors LoRA) + optional latency stats.",
    )
    ap.add_argument(
        "--image",
        "-i",
        default=None,
        help="Single image path. If omitted, the first on-disk row from --split is used.",
    )
    ap.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="valid",
        help="CSV to sample from when --image is not set (default: valid).",
    )
    ap.add_argument(
        "--instruction",
        default=FIXED_INSTRUCTION,
        help=f"Matches training fixed instruction (default: {FIXED_INSTRUCTION!r}).",
    )
    ap.add_argument(
        "--adapter",
        "-a",
        default=OUTPUT_DIR,
        help=f"LoRA folder (adapter_model.safetensors). Default: {OUTPUT_DIR}",
    )
    ap.add_argument(
        "--model",
        "-m",
        default=None,
        help="Overrides --adapter when set (directory with captioning LoRA).",
    )
    ap.add_argument(
        "--base-model",
        "-b",
        default=MODEL_ID,
        help=f"Hugging Face base id (default: {MODEL_ID}).",
    )
    ap.add_argument(
        "--base-only",
        action="store_true",
        help="Load base SmolVLM only (no LoRA).",
    )
    ap.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA into base after load (more VRAM, often faster).",
    )
    ap.add_argument(
        "--limit",
        "-l",
        type=int,
        default=10,
        help="Number of timed inference runs after warmup.",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations before timing (default: 3).",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="When sampling from CSV: shuffle and run this many distinct images (default: 1). Ignored if --image is set.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Shuffle seed for CSV sampling.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation length (default: 256).",
    )
    ap.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only print latency stats for one image (no per-sample captions).",
    )
    return ap.parse_args()


def resolve_images(args) -> list[tuple[str, str | None]]:
    """Returns list of (image_path, reference_caption_or_none)."""
    if args.image:
        p = os.path.abspath(args.image)
        if not os.path.isfile(p):
            raise SystemExit(f"Image not found: {p}")
        return [(p, None)]

    csv_path = os.path.join(DATA_ROOT, f"{args.split}.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Missing {csv_path}")

    random.seed(args.seed)
    records = load_split(csv_path, DATA_ROOT)
    if not records:
        raise SystemExit(
            f"No CSV rows with existing files under {DATA_ROOT}. "
            "Sync images so paths in the CSV resolve."
        )
    random.shuffle(records)
    out = []
    for rec in records[: args.num_samples]:
        ref = rec["captions"][0] if rec["captions"] else None
        out.append((rec["image"], ref))
    return out


def main():
    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype = torch.float16 if device != "cpu" else torch.float32

    adapter_arg = args.model if args.model else args.adapter
    adapter_dir = None if args.base_only else os.path.abspath(adapter_arg)
    if adapter_dir and not os.path.isdir(adapter_dir):
        print(f"Warning: adapter path is not a directory: {adapter_dir}")
        adapter_dir = None

    pairs = resolve_images(args)

    mdl = SatelliteCaptionModel(
        base_model_id=args.base_model,
        adapter_dir=adapter_dir,
        merge_lora=args.merge_lora,
        device=device,
        dtype=dtype,
    )

    bench_path = pairs[0][0]

    def run_caption(path: str) -> str:
        return mdl.caption(path, args.instruction, args.max_new_tokens)

    last: str | None = None
    if not args.benchmark_only:
        print(f"DATA_ROOT: {DATA_ROOT}")
        for path, ref in pairs:
            cap = run_caption(path)
            last = cap
            print(f"\n--- {path} ---")
            if ref is not None:
                print(f"Reference (first): {ref[:300]}{'...' if len(ref) > 300 else ''}")
            print(f"Predicted: {cap}")
    else:
        last = run_caption(bench_path)

    for _ in range(max(0, args.warmup)):
        last = run_caption(bench_path)

    times = []
    for _ in range(args.limit):
        start = time.perf_counter()
        last = run_caption(bench_path)
        times.append(time.perf_counter() - start)

    print("\nMean latency:", float(np.mean(times)))
    print("Std latency:", float(np.std(times)))
    print("Last caption:", last)


if __name__ == "__main__":
    main()
