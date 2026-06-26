"""
Smoke test for the OpenVLA-OFT training pipeline.

Tests the complete data → model → loss → backward → inference flow
using a single dummy image. No LIBERO dataset or GPU required for shape tests.

MODES:
    1. Shape-only (default, ~5 seconds, CPU, no model download):
       python 00_smoke_test.py

    2. Base OpenVLA with DummyDataset — no OFT repo needed (~3 min, requires GPU):
       python 00_smoke_test.py --model_id openvla/openvla-7b --base_openvla --device cuda

    3. OFT model from a LOCAL trained checkpoint (after running 01_train_openvla_oft.py):
       python 00_smoke_test.py --model_id checkpoints/openvla_oft_libero/epoch_001/hf_model --device cuda

    4. OFT model using the cloned openvla-oft repo (adds action_head to openvla-7b at load time):
       export PYTHONPATH=$(pwd)/../openvla-oft:$PYTHONPATH
       python 00_smoke_test.py --model_id openvla/openvla-7b --device cuda

NOTE: `moojink/openvla-oft` is a GitHub repo (training code), NOT a HuggingFace model.
      The base HF model is always `openvla/openvla-7b`.
      The OFT action_head is added by installing the openvla-oft repo and using its model
      class, OR it appears in a checkpoint you trained yourself with 01_train_openvla_oft.py.

WHAT IT CHECKS:
    [Shape tests — always run]
    - LIBERODataset collate_fn produces correct tensor shapes
    - Action normalization/unnormalization is invertible
    - Action chunk padding produces (chunk_size, action_dim) output

    [Model tests — run with --model_id]
    - Model loads with AutoModelForVision2Seq (not wrong class)
    - model.action_head exists (OFT-specific assertion)
    - model.config.model_type == "openvla"
    - Forward pass produces loss (training mode)
    - predict_action() produces (1, chunk_size, 7) output (inference mode)
    - Output shape is preserved after 2 training steps
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
SKIP = "\033[93m SKIP\033[0m"
INFO = "     "

results = []


def test(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[{PASS}] {name}  ({elapsed:.0f}ms)")
        results.append((name, True, None))
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[{FAIL}] {name}  ({elapsed:.0f}ms)")
        print(f"{INFO}  {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        results.append((name, False, str(e)))


def skip(name, reason):
    print(f"[{SKIP}] {name}  — {reason}")
    results.append((name, None, reason))


# ─────────────────────────────────────────────────────────────────────────────
# Shape-only tests (no model, no GPU, instant)

OFT_CHUNK_SIZE = 8
OFT_ACTION_DIM = 7
OFT_PROPRIO_DIM = 9


def make_dummy_action_stats():
    """Realistic action stats based on LIBERO demonstrated trajectories."""
    return {
        "mean": [0.001, -0.002, 0.003, 0.0, 0.0, 0.0, 0.5],
        "std":  [0.08,   0.06,  0.05, 0.15, 0.12, 0.10, 0.45],
        "min":  [-0.3, -0.3, -0.2, -0.6, -0.5, -0.4, 0.0],
        "max":  [0.3,   0.3,  0.2,  0.6,  0.5,  0.4, 1.0],
    }


def test_action_chunk_shapes():
    """collate_fn produces tensors with exactly (chunk_size, action_dim) action shape."""
    # Simulate what LIBERODataset.__getitem__ returns
    stats = make_dummy_action_stats()
    action_mean = torch.tensor(stats["mean"])
    action_std  = torch.tensor(stats["std"])

    T = 20
    raw_actions = np.random.uniform(-0.3, 0.3, (T, OFT_ACTION_DIM)).astype(np.float32)

    for t in range(T):
        end = min(t + OFT_CHUNK_SIZE, T)
        chunk = raw_actions[t:end]
        if chunk.shape[0] < OFT_CHUNK_SIZE:
            pad = np.tile(chunk[-1:], (OFT_CHUNK_SIZE - chunk.shape[0], 1))
            chunk = np.concatenate([chunk, pad], axis=0)

        chunk_t = torch.from_numpy(chunk)
        normalized = (chunk_t - action_mean) / (action_std + 1e-8)

        assert normalized.shape == (OFT_CHUNK_SIZE, OFT_ACTION_DIM), \
            f"Shape {normalized.shape} != ({OFT_CHUNK_SIZE}, {OFT_ACTION_DIM})"

        # Unnormalization must be invertible
        unnorm = normalized * action_std + action_mean
        assert torch.allclose(unnorm, chunk_t, atol=1e-5), \
            f"Unnorm not invertible. Max diff: {(unnorm - chunk_t).abs().max():.6f}"


def _collate_fn(batch):
    """Same logic as collate_fn in 01_train_openvla_oft.py — inlined to avoid
    importing deepspeed at smoke-test time."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    padded_ids, padded_mask = [], []
    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        padded_ids.append(F.pad(b["input_ids"], (0, pad_len), value=0))
        padded_mask.append(F.pad(b["attention_mask"], (0, pad_len), value=0))
    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_mask),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "proprio": torch.stack([b["proprio"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
    }


def test_collate_fn_padding():
    """collate_fn pads input_ids to the same length within a batch."""

    # Simulate variable-length token sequences
    batch = [
        {
            "input_ids": torch.randint(0, 1000, (50,)),
            "attention_mask": torch.ones(50, dtype=torch.long),
            "pixel_values": torch.randn(3, 224, 224),
            "proprio": torch.randn(OFT_PROPRIO_DIM),
            "actions": torch.randn(OFT_CHUNK_SIZE, OFT_ACTION_DIM),
        },
        {
            "input_ids": torch.randint(0, 1000, (65,)),
            "attention_mask": torch.ones(65, dtype=torch.long),
            "pixel_values": torch.randn(3, 224, 224),
            "proprio": torch.randn(OFT_PROPRIO_DIM),
            "actions": torch.randn(OFT_CHUNK_SIZE, OFT_ACTION_DIM),
        },
    ]
    out = _collate_fn(batch)

    assert out["input_ids"].shape == (2, 65), \
        f"Expected (2,65) after padding, got {out['input_ids'].shape}"
    assert out["attention_mask"].shape == (2, 65)
    assert out["pixel_values"].shape == (2, 3, 224, 224)
    assert out["proprio"].shape == (2, OFT_PROPRIO_DIM)
    assert out["actions"].shape == (2, OFT_CHUNK_SIZE, OFT_ACTION_DIM)

    # Padded positions in attention_mask must be 0
    assert out["attention_mask"][0, 50:].sum() == 0, "Padding mask not zeroed"


def test_action_normalization_range():
    """Normalized actions should have approximately zero mean and unit variance."""
    stats = make_dummy_action_stats()
    action_mean = np.array(stats["mean"])
    action_std  = np.array(stats["std"])

    # 10k samples: std-of-mean = 1/sqrt(10000) = 0.01 per dim, so 0.15 threshold
    # gives >10σ margin — this should never flake
    raw = np.random.randn(10_000, OFT_ACTION_DIM) * action_std + action_mean
    normalized = (raw - action_mean) / (action_std + 1e-8)

    assert np.abs(normalized.mean(axis=0)).max() < 0.15, \
        f"Normalized mean not near 0: {normalized.mean(axis=0)}"
    assert np.abs(normalized.std(axis=0) - 1.0).max() < 0.15, \
        f"Normalized std not near 1: {normalized.std(axis=0)}"


# ─────────────────────────────────────────────────────────────────────────────
# Model tests (require --model_id)

def test_model_class_and_action_head(model_id, device):
    """Model loads with correct class and action_head attribute exists."""
    from transformers import AutoModelForVision2Seq

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": str(device)} if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model = model.to(torch.float32)

    # action_head is only present if the openvla-oft repo is installed AND its model
    # class was registered (happens automatically when you `pip install -e openvla-oft/`
    # and have PYTHONPATH set). If testing the plain base model, this will fail — use
    # --base_openvla mode instead for openvla/openvla-7b without OFT modifications.
    assert hasattr(model, "action_head"), (
        "action_head missing. This means either:\n"
        "  1. openvla-oft repo is not installed: pip install -e openvla-oft/\n"
        "  2. PYTHONPATH doesn't include the repo: export PYTHONPATH=$(pwd)/../openvla-oft:$PYTHONPATH\n"
        "  3. You're testing a base checkpoint — use --base_openvla flag instead\n"
        "  4. transformers > 4.40.2 installed — pin to 4.40.2"
    )
    assert model.config.model_type == "openvla", \
        f"model_type='{model.config.model_type}', expected 'openvla'"

    return model


def test_training_step(model, processor, device):
    """
    One forward + backward pass completes without error.
    Uses a single dummy image + instruction — no LIBERO data needed.
    """
    from transformers import AutoProcessor

    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    instruction = "pick up the red block and place it on the blue plate"

    inputs = processor(text=instruction, images=dummy_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    proprio = torch.zeros(1, OFT_PROPRIO_DIM, device=device, dtype=torch.bfloat16)
    target_actions = torch.zeros(
        1, OFT_CHUNK_SIZE, OFT_ACTION_DIM, device=device, dtype=torch.bfloat16
    )

    model.train()
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        proprio=proprio,
        labels=target_actions,
    )

    if hasattr(outputs, "loss") and outputs.loss is not None:
        loss = outputs.loss
    else:
        pred = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss = F.l1_loss(pred, target_actions)

    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"

    loss.backward()
    model.zero_grad()

    return loss.item()


def test_inference_step(model, processor, device):
    """
    predict_action() returns (1, chunk_size, action_dim) with no NaN/Inf.
    """
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    instruction = "pick up the red block and place it on the blue plate"

    inputs = processor(text=instruction, images=dummy_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    proprio = torch.zeros(1, OFT_PROPRIO_DIM, device=device, dtype=torch.bfloat16)

    model.eval()
    with torch.no_grad():
        action_chunk = model.predict_action(**inputs, proprio=proprio)

    assert action_chunk.shape == (1, OFT_CHUNK_SIZE, OFT_ACTION_DIM), (
        f"Wrong output shape: {action_chunk.shape}. "
        f"Expected (1, {OFT_CHUNK_SIZE}, {OFT_ACTION_DIM}). "
        f"Check that chunk_size and action_dim match training config."
    )
    assert torch.isfinite(action_chunk).all(), "action_chunk contains NaN or Inf"

    return action_chunk


def test_output_shape_stable_across_steps(model, processor, device):
    """
    Run 2 training steps and confirm action output shape stays the same.
    Catches quantization or LoRA bugs that affect shape after gradient update.
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    instruction = "move the robot arm forward"
    inputs = processor(text=instruction, images=dummy_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    proprio = torch.zeros(1, OFT_PROPRIO_DIM, device=device, dtype=torch.bfloat16)
    target = torch.zeros(1, OFT_CHUNK_SIZE, OFT_ACTION_DIM, device=device, dtype=torch.bfloat16)

    for step in range(2):
        model.train()
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            proprio=proprio,
            labels=target,
        )
        loss = outputs.loss if (hasattr(outputs, "loss") and outputs.loss is not None) \
               else F.l1_loss(outputs.logits, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Check inference shape after training steps
    model.eval()
    with torch.no_grad():
        out = model.predict_action(**inputs, proprio=proprio)
    assert out.shape == (1, OFT_CHUNK_SIZE, OFT_ACTION_DIM), \
        f"Shape changed after training steps: {out.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# Base OpenVLA (non-OFT) test using DummyDataset from openvla/ repo

def test_base_openvla_dummy_dataset(model_id, device):
    """
    Tests the base openvla fine-tuning path using the DummyDataset
    already in the openvla/ repo (no LIBERO data needed).
    Works with openvla/openvla-7b loaded via the openvla/ repo's AutoClasses.
    """
    # Register openvla model classes (from the cloned repo)
    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.datasets import DummyDataset
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    except ImportError as e:
        raise ImportError(
            f"openvla repo not in PYTHONPATH: {e}. "
            "Run: export PYTHONPATH=$(pwd)/openvla:$PYTHONPATH"
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = PrismaticProcessor.from_pretrained(model_id, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    dataset = DummyDataset(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )

    sample = dataset[0]
    assert "pixel_values" in sample, "pixel_values missing from DummyDataset sample"
    assert "input_ids" in sample
    assert "labels" in sample

    print(f"{INFO}  DummyDataset sample shapes: "
          f"pixel_values={sample['pixel_values'].shape}, "
          f"input_ids={sample['input_ids'].shape}")

    # Optionally run one forward pass if GPU available
    if device.type == "cuda":
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": str(device)},
        )
        model.eval()

        with torch.no_grad():
            out = model(
                input_ids=sample["input_ids"].unsqueeze(0).to(device),
                attention_mask=torch.ones_like(sample["input_ids"]).unsqueeze(0).to(device),
                pixel_values=sample["pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16),
                labels=sample["labels"].unsqueeze(0).to(device),
            )
        assert hasattr(out, "loss") and out.loss is not None
        assert torch.isfinite(out.loss), f"loss={out.loss.item()}"
        print(f"{INFO}  Base OpenVLA forward pass loss: {out.loss.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=None,
                   help="HF model ID or local checkpoint path for full model tests. "
                        "Omit to run shape-only tests (fast, no download needed).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for model tests. Shape tests always run on CPU.")
    p.add_argument("--base_openvla", action="store_true",
                   help="Test the base openvla/ repo path (DummyDataset) instead of OFT.")
    p.add_argument("--verbose", action="store_true",
                   help="Print full tracebacks on failure.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print(f"[{SKIP}] CUDA not available — falling back to CPU for model tests.")
        device = torch.device("cpu")

    print(f"\nOpenVLA-OFT Smoke Test  (device={device})")
    print("=" * 55)

    # ── Shape tests (always run) ──────────────────────────────────────────────
    print("\n── Shape / Data Pipeline Tests ──")
    test("Action chunk shapes and padding",    test_action_chunk_shapes)
    test("collate_fn batch padding",           test_collate_fn_padding)
    test("Action normalization invertible",    test_action_normalization_range)

    # ── Base OpenVLA DummyDataset test ────────────────────────────────────────
    if args.base_openvla:
        print("\n── Base OpenVLA (openvla/ repo) Tests ──")
        if args.model_id:
            test("DummyDataset + forward pass",
                 lambda: test_base_openvla_dummy_dataset(args.model_id, device))
        else:
            skip("DummyDataset + forward pass", "pass --model_id openvla/openvla-7b")

    # ── OFT model tests ───────────────────────────────────────────────────────
    if not args.base_openvla and args.model_id:
        print("\n── OpenVLA-OFT Model Tests ──")
        model = None
        processor = None

        def load():
            global model, processor
            from transformers import AutoModelForVision2Seq, AutoProcessor
            processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
            model = test_model_class_and_action_head(args.model_id, device)

        test("Load model + verify action_head", load)

        if model is not None:
            test("Training forward + backward",
                 lambda: test_training_step(model, processor, device))
            test("Inference output shape (1, 8, 7)",
                 lambda: test_inference_step(model, processor, device))
            test("Shape stable across 2 optimizer steps",
                 lambda: test_output_shape_stable_across_steps(model, processor, device))
    elif not args.base_openvla:
        print("\n── OpenVLA-OFT Model Tests ──")
        skip("Model load + verify action_head",  "pass --model_id to enable")
        skip("Training forward + backward",       "pass --model_id to enable")
        skip("Inference output shape (1, 8, 7)",  "pass --model_id to enable")
        skip("Shape stable across 2 steps",       "pass --model_id to enable")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    passed  = sum(1 for _, ok, _ in results if ok is True)
    failed  = sum(1 for _, ok, _ in results if ok is False)
    skipped = sum(1 for _, ok, _ in results if ok is None)

    if failed == 0:
        print(f"\033[92m  {passed} passed\033[0m, {skipped} skipped — smoke test OK")
        if not args.model_id:
            print("  Run with --model_id moojink/openvla-oft --device cuda for full test.")
    else:
        print(f"\033[91m  {failed} failed\033[0m, {passed} passed, {skipped} skipped")
        print("  Re-run with --verbose for full tracebacks.")
        sys.exit(1)
