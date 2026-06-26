"""
Fine-tune OpenVLA-OFT on all 4 LIBERO task suites using DeepSpeed ZeRO-2 on 8x H100.

ARCHITECTURE NOTE:
    In openvla-oft, the action_head is a *separate* nn.Module (L1RegressionActionHead),
    NOT stored as self.action_head on the VLM. This script:
      1. Loads openvla/openvla-7b as the base VLM (AutoModelForVision2Seq).
      2. Creates L1RegressionActionHead separately, attaches it to the model BEFORE
         DeepSpeed init so ZeRO-2 manages its parameters together with the VLM.
      3. Forward pass uses output_hidden_states=True; action loss is computed by
         extracting hidden states at action-token positions and passing to action_head.

SETUP (run once before this script):
    # 1. Clone openvla-oft and install it
    git clone https://github.com/moojink/openvla-oft.git
    pip install -e openvla-oft/

    # 2. Install LIBERO
    pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git@master

    # 3. Download LIBERO datasets
    bash openvla_oft_pipeline/download_libero.sh

    # 4. Set environment variables
    export MUJOCO_GL=egl
    export PYTHONPATH="$(pwd)/openvla-oft:$PYTHONPATH"
    export WANDB_MODE=offline

LAUNCH (do not run this file directly — use launch_train.sh):
    bash launch_train.sh
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import deepspeed
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

# openvla-oft must be on PYTHONPATH (set by launch_train.sh via openvla-oft/ directory)
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
)

# ─────────────────────────────────────────────────────────────────────────────
# LIBERO task suite definitions — all 4 evaluated per the OFT paper
LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]

# OFT paper config for LIBERO (Table 1 in the OFT paper)
OFT_CHUNK_SIZE  = 8    # action chunk length: 8 future steps
OFT_ACTION_DIM  = 7    # [dx, dy, dz, droll, dpitch, dyaw, gripper]
OFT_PROPRIO_DIM = 8    # eef_pos(3) + eef_quat(4) + gripper_qpos(1)
OFT_IMAGE_SIZE  = 224


# ─────────────────────────────────────────────────────────────────────────────
# Dataset

class LIBERODataset(Dataset):
    """
    Reads LIBERO HDF5 demos and returns batches compatible with the OFT training loop.

    Each sample contains:
      input_ids      — tokenized prompt WITH action tokens embedded (text only, no image)
      attention_mask — corresponding mask
      pixel_values   — preprocessed image tensor
      labels         — same shape as input_ids; IGNORE_INDEX for non-action positions,
                       discretized action token IDs at action positions
      actions        — normalized continuous action chunk (chunk_size, action_dim)
                       used as the L1 regression target
    """

    def __init__(
        self,
        suite_names: list,
        data_root: str,
        processor,
        action_tokenizer,
        action_stats: dict,
        chunk_size: int = OFT_CHUNK_SIZE,
        augment: bool = True,
    ):
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.chunk_size = chunk_size
        self.augment = augment
        self.action_mean = torch.tensor(action_stats["mean"], dtype=torch.float32)
        self.action_std  = torch.tensor(action_stats["std"],  dtype=torch.float32)

        # ── Placeholder action token IDs fed to the model as INPUT ─────────────
        # CRITICAL: identical to the placeholder used at eval time (02_eval_libero.py).
        # The model must see the SAME action-token sequence during training as during
        # inference. Feeding the ground-truth action tokens here instead would let the
        # action_head trivially decode its own input embeddings (teacher-forcing leak)
        # and never learn vision→action — at eval those tokens are zeros, so the head
        # emits a constant action and every rollout fails (0% success). GT actions are
        # used ONLY as the L1 target (see `actions` in __getitem__), never as input.
        #
        # We build the action token IDs DIRECTLY, not by decoding to text and
        # re-tokenizing: ActionTokenizer.__call__ returns a *decoded string*, and that
        # ID→text→ID round-trip is not reversible for these special tokens — it turned
        # a 56-action chunk into 57 tokens and silently misaligned the -56: hidden-state
        # slice. Mirror this exact construction in 02_eval_libero.py.
        zero_chunk = np.zeros((self.chunk_size, OFT_ACTION_DIM), dtype=np.float32)
        disc = np.digitize(np.clip(zero_chunk, -1.0, 1.0), action_tokenizer.bins)
        self.action_token_ids = torch.from_numpy(
            (processor.tokenizer.vocab_size - disc).reshape(-1).astype(np.int64)
        )  # (chunk_size * OFT_ACTION_DIM,) == (56,)
        assert self.action_token_ids.numel() == self.chunk_size * OFT_ACTION_DIM

        self.samples = []  # (hdf5_path, demo_key, timestep, T, instruction)
        for suite_name in suite_names:
            suite_data_dir = Path(data_root) / suite_name
            # libero_long stores HDF5s in subdirectories; others store directly
            hdf5_files = sorted(suite_data_dir.glob("*.hdf5")) or \
                         sorted(suite_data_dir.glob("**/*.hdf5"))
            if not hdf5_files:
                raise FileNotFoundError(
                    f"No .hdf5 files found in {suite_data_dir}. "
                    "Check DATA_ROOT and re-run the dataset download."
                )
            for hdf5_path in hdf5_files:
                with h5py.File(hdf5_path, "r") as f:
                    # The language instruction lives in data.attrs["problem_info"] as a
                    # JSON string (key "language_instruction") — NOT in the root f.attrs
                    # (which is empty). Reading the wrong place gave every sample an empty
                    # instruction, so the model never learned language conditioning and
                    # could not tell the 10 tasks in a suite apart at eval time.
                    problem_info = f["data"].attrs["problem_info"]
                    if isinstance(problem_info, bytes):
                        problem_info = problem_info.decode()
                    instruction = json.loads(problem_info)["language_instruction"]
                    for demo_key in f["data"].keys():
                        T = f["data"][demo_key]["actions"].shape[0]
                        for t in range(T):
                            self.samples.append((str(hdf5_path), demo_key, t, T, instruction))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdf5_path, demo_key, t, T, instruction = self.samples[idx]

        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]
            # Rotate 180° to match the eval/sim convention. The raw HDF5 stores
            # agentview_rgb in OpenGL convention (macros_image_convention="opengl",
            # i.e. upside-down). 02_eval_libero.py and the reference run_libero_eval.py
            # both apply [::-1, ::-1] to the sim's agentview_image. Training MUST use
            # the same orientation or the model sees flipped images at eval (→ 0%).
            image_np = np.ascontiguousarray(
                demo["obs"]["agentview_rgb"][t][::-1, ::-1]
            )  # (H, W, 3) uint8

            # Proprio: ee_pos(3) + ee_ori(3) + gripper_states(2) = 8D
            ee_pos        = demo["obs"]["ee_pos"][t]           # (3,)
            ee_ori        = demo["obs"]["ee_ori"][t]           # (3,) Euler angles
            gripper       = demo["obs"]["gripper_states"][t]   # (2,)
            proprio       = np.concatenate([ee_pos, ee_ori, gripper]).astype(np.float32)

            # Raw action chunk (BEFORE normalization — ActionTokenizer does its own binning)
            end = min(t + self.chunk_size, T)
            raw_actions = demo["actions"][t:end].astype(np.float32)  # (≤chunk, 7)
            if raw_actions.shape[0] < self.chunk_size:
                pad = np.tile(raw_actions[-1:], (self.chunk_size - raw_actions.shape[0], 1))
                raw_actions = np.concatenate([raw_actions, pad], axis=0)  # (chunk, 7)

        # ── Normalized actions for L1 loss target ─────────────────────────────
        actions = torch.from_numpy(raw_actions)
        actions = (actions - self.action_mean) / (self.action_std + 1e-8)

        # ── Build prompt prefix (Vicuna-v1.5 format used by openvla) ──────────
        # Everything up to and including "ASSISTANT: "; the action tokens are
        # appended as raw IDs afterwards so we never round-trip them through text.
        sys_msg = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        )
        prefix = (
            f"{sys_msg}"
            f"USER: What action should the robot take to {instruction}? "
            f"ASSISTANT: "
        )

        # Tokenize text prefix (no image tokens in input_ids — pixel_values handled
        # separately; the model prepends vision patches internally).
        prefix_ids = self.processor.tokenizer(
            prefix, add_special_tokens=True, return_tensors="pt"
        )["input_ids"].squeeze(0)

        # Append the PLACEHOLDER action token IDs (identical at train & eval), then EOS.
        # The real actions live only in `actions` (the L1 target) above.
        eos_id = torch.tensor([self.processor.tokenizer.eos_token_id], dtype=torch.long)
        input_ids = torch.cat([prefix_ids, self.action_token_ids, eos_id])
        attention_mask = torch.ones_like(input_ids)

        # ── Build labels: IGNORE_INDEX except for action token positions ───────
        # Action tokens are the last (OFT_ACTION_DIM * OFT_CHUNK_SIZE + 1) tokens before EOS.
        # "+1" accounts for EOS which we keep as a target token.
        labels = input_ids.clone()
        action_chunk_len = OFT_ACTION_DIM * OFT_CHUNK_SIZE  # = 56
        # Mask everything except the action chunk + EOS
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX

        # ── Process image ──────────────────────────────────────────────────────
        image_pil = Image.fromarray(image_np)
        if self.augment:
            import torchvision.transforms.functional as TF
            import random
            if random.random() > 0.5:
                image_pil = TF.adjust_brightness(image_pil, brightness_factor=random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image_pil = TF.adjust_contrast(image_pil, contrast_factor=random.uniform(0.8, 1.2))

        pixel_values = self.processor.image_processor(
            images=image_pil, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "labels":         labels,
            "proprio":        torch.from_numpy(proprio),
            "actions":        actions,   # (chunk_size, action_dim) normalized
        }


def collate_fn(batch):
    """Pad variable-length input_ids / attention_mask / labels to the same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    padded_ids, padded_mask, padded_labels = [], [], []
    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        padded_ids.append(F.pad(b["input_ids"],      (0, pad_len), value=0))
        padded_mask.append(F.pad(b["attention_mask"], (0, pad_len), value=0))
        padded_labels.append(F.pad(b["labels"],       (0, pad_len), value=IGNORE_INDEX))

    return {
        "input_ids":      torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_mask),
        "pixel_values":   torch.stack([b["pixel_values"] for b in batch]),
        "labels":         torch.stack(padded_labels),
        "proprio":        torch.stack([b["proprio"]  for b in batch]),
        "actions":        torch.stack([b["actions"]  for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Action statistics

def compute_action_stats(suite_names: list, data_root: str) -> dict:
    """Precompute per-dimension mean / std over all training demos (first run)."""
    all_actions = []
    for suite_name in suite_names:
        suite_dir = Path(data_root) / suite_name
        hdf5_list = sorted(suite_dir.glob("*.hdf5")) or sorted(suite_dir.glob("**/*.hdf5"))
        for hdf5_path in hdf5_list:
            with h5py.File(hdf5_path, "r") as f:
                for demo_key in f["data"].keys():
                    all_actions.append(f["data"][demo_key]["actions"][:])

    actions = np.concatenate(all_actions, axis=0)  # (N_total, 7)
    return {
        "mean": actions.mean(axis=0).tolist(),
        "std":  actions.std(axis=0).tolist(),
        "min":  actions.min(axis=0).tolist(),
        "max":  actions.max(axis=0).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading

def load_oft_model(model_id: str, local_rank: int):
    """
    Load OpenVLA base model (openvla/openvla-7b) and its processor.

    IMPORTANT: action_head is NOT a model attribute — it is a separate
    L1RegressionActionHead created in train() and attached before DeepSpeed init
    so ZeRO-2 shards its parameters alongside the VLM.
    """
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # NO device_map — DeepSpeed owns device placement
    )

    assert model.config.model_type == "openvla", (
        f"model_type='{model.config.model_type}', expected 'openvla'."
    )

    if local_rank == 0:
        total = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"[Model] Loaded: {model_id}  ({total:.2f}B params)")

    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# Training

def train(args):
    # ── DeepSpeed init ────────────────────────────────────────────────────────
    # Explicit 60-min collective timeout — the default 10-min NCCL watchdog killed
    # a previous run when one rank stalled on slow HDF5 reads during an allreduce.
    from datetime import timedelta
    deepspeed.init_distributed(timeout=timedelta(minutes=60))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    is_main = (local_rank == 0)

    # Gloo (CPU) group for synchronization around rank-0-only work (HF export,
    # quick eval). NCCL barriers are watchdog-killed after 10 min — and the ZeRO-2
    # gradient allreduce runs on an internal subgroup that ignores the timeout
    # passed to init_distributed — so slow rank-0 work must never leave the other
    # ranks sitting in a NCCL collective.
    gloo_group = torch.distributed.new_group(
        backend="gloo", timeout=timedelta(hours=3)
    )

    if is_main:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "openvla-oft-libero"),
            name=args.run_name,
            config=vars(args),
        )

    # ── Action statistics ─────────────────────────────────────────────────────
    stats_path = Path(args.output_dir) / "action_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            action_stats = json.load(f)
        if is_main:
            print(f"[Data] Loaded action stats from {stats_path}")
    else:
        if is_main:
            print("[Data] Computing action statistics (first run only)...")
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            action_stats = compute_action_stats(LIBERO_SUITES, args.data_root)
            with open(stats_path, "w") as f:
                json.dump(action_stats, f, indent=2)
            print(f"[Data] Action stats saved to {stats_path}")
        torch.distributed.barrier()
        with open(stats_path) as f:
            action_stats = json.load(f)

    # ── Load model ────────────────────────────────────────────────────────────
    processor, model = load_oft_model(args.model_id, local_rank)

    # ── Create L1RegressionActionHead and attach to model ────────────────────
    # Must happen BEFORE deepspeed.initialize() so ZeRO-2 manages its params.
    llm_dim = model.language_model.config.hidden_size  # 4096 for LLaMA-2-7B
    action_head = L1RegressionActionHead(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=OFT_ACTION_DIM,
    ).to(torch.bfloat16)
    model.action_head = action_head  # attach so DeepSpeed sees it

    # Gradient checkpointing: recomputes activations during backward instead of storing
    # them — cuts activation memory ~50% at the cost of ~30% more compute.
    model.gradient_checkpointing_enable()

    if is_main:
        head_params = sum(p.numel() for p in action_head.parameters()) / 1e6
        print(f"[Model] action_head: L1RegressionActionHead ({head_params:.1f}M params, llm_dim={llm_dim})")

    # ── Warm restart from a previous (possibly partial) checkpoint ────────────
    # Recover from a crashed run whose DeepSpeed optimizer shards were lost/corrupt.
    # We load ONLY the model weights (the unsharded `module` state dict — ZeRO-2 does
    # not shard parameters) into the model BEFORE deepspeed.initialize(), then start a
    # FRESH optimizer/scheduler. This keeps every trained weight while sidestepping a
    # broken optimizer resume; only Adam momentum is lost (re-warms in a few hundred
    # steps). Accepts either a raw state dict or a DeepSpeed `mp_rank_*_model_states.pt`.
    if args.init_weights:
        ckpt = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        state_dict = ckpt["module"] if isinstance(ckpt, dict) and "module" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main:
            n_loaded = len(state_dict)
            print(f"[InitWeights] Loaded {n_loaded} tensors from {args.init_weights}")
            print(f"[InitWeights] missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"[InitWeights]   e.g. missing: {list(missing)[:5]}")
            if unexpected:
                print(f"[InitWeights]   e.g. unexpected: {list(unexpected)[:5]}")
            # Sanity guard: a near-total mismatch means the wrong file / arch.
            if len(missing) > n_loaded * 0.5:
                raise RuntimeError(
                    "init_weights matched <50% of model params — wrong checkpoint or arch."
                )
        del ckpt, state_dict

    # ── ActionTokenizer ───────────────────────────────────────────────────────
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    dataset = LIBERODataset(
        suite_names=LIBERO_SUITES,
        data_root=args.data_root,
        processor=processor,
        action_tokenizer=action_tokenizer,
        action_stats=action_stats,
        chunk_size=OFT_CHUNK_SIZE,
        augment=args.augment,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    loader  = DataLoader(
        dataset,
        batch_size=args.per_gpu_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Cap batches per epoch — at ~1.3 it/s a full pass over all 1M samples takes
    # ~6.7 h; --max_steps_per_epoch trades coverage for wall time. The sampler
    # reshuffles each epoch, so capped epochs still see fresh data.
    steps_per_epoch = len(loader)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)

    if is_main:
        print(f"[Data] Total samples: {len(dataset)}, batches/epoch: {len(loader)}, "
              f"steps/epoch: {steps_per_epoch}")

    # ── DeepSpeed engine ──────────────────────────────────────────────────────
    ds_config = {
        "train_micro_batch_size_per_gpu": args.per_gpu_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "train_batch_size": args.per_gpu_batch_size * args.grad_accum * world_size,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps,
                # Only the epochs we actually train this run (start_epoch..num_epochs).
                "total_num_steps": (args.num_epochs - args.start_epoch) * steps_per_epoch,
            },
        },
    }
    with open(args.ds_config) as f:
        static_config = json.load(f)
    for k, v in static_config.items():
        if k not in ("_comment",) and k not in ds_config:
            ds_config[k] = v

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(args.start_epoch, args.num_epochs):
        sampler.set_epoch(epoch)
        model_engine.train()

        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(
            tqdm(loader, total=steps_per_epoch, disable=not is_main, desc=f"Epoch {epoch+1}")
        ):
            if batch_idx >= steps_per_epoch:
                break
            device = model_engine.device

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            labels         = batch["labels"].to(device)
            target_actions = batch["actions"].to(device, dtype=torch.bfloat16)  # (B, chunk, 7)
            B = input_ids.shape[0]

            # Forward pass — need hidden states to feed to action_head
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,            # tokenized action labels (for action-pos masking)
                output_hidden_states=True,
            )

            # Extract hidden states at action-token positions.
            # The full sequence in hidden states = [vision_patches | text_tokens | EOS].
            # Action tokens are the last OFT_CHUNK_SIZE * OFT_ACTION_DIM tokens before EOS.
            action_len = OFT_CHUNK_SIZE * OFT_ACTION_DIM   # = 56
            last_hidden = outputs.hidden_states[-1]          # (B, total_len, D)
            # Slice: skip EOS (-1), take action_len positions before it
            action_hidden = last_hidden[:, -(action_len + 1):-1, :]  # (B, 56, D)

            # Predict actions via action_head: (B, 56, D) → (B, 8, 7)
            predicted_actions = model_engine.module.action_head.predict_action(action_hidden)

            # L1 regression loss against normalized ground-truth actions
            loss = F.l1_loss(predicted_actions, target_actions)

            model_engine.backward(loss)
            model_engine.step()

            epoch_loss  += loss.item()
            global_step += 1

            if is_main and global_step % args.log_every == 0:
                elapsed = time.time() - t0
                mem_stats = {
                    f"gpu_{i}_mem_mb": torch.cuda.memory_allocated(i) / 1e6
                    for i in range(torch.cuda.device_count())
                }
                wandb.log({
                    "train/loss":          loss.item(),
                    "train/epoch":         epoch + batch_idx / steps_per_epoch,
                    "train/global_step":   global_step,
                    "train/lr":            optimizer.param_groups[0]["lr"],
                    "perf/steps_per_sec":  args.log_every / elapsed,
                    **mem_stats,
                })
                t0 = time.time()

        # ── Checkpoint ────────────────────────────────────────────────────────
        # save_checkpoint() is a COLLECTIVE — every rank must call it (ZeRO-2
        # shards optimizer state across ranks). Calling it under is_main
        # deadlocks rank 0 while ranks 1-7 run ahead into the next epoch.
        ckpt_dir = Path(args.output_dir) / f"epoch_{epoch+1:03d}"
        model_engine.save_checkpoint(str(ckpt_dir))
        if is_main:
            # Guard the rank-0 HF export: a disk-full / IO error here must not crash
            # the run (the DeepSpeed checkpoint above already holds the weights). The
            # original run died exactly here when `/` filled during save_pretrained.
            try:
                model_engine.module.save_pretrained(str(ckpt_dir / "hf_model"))
                processor.save_pretrained(str(ckpt_dir / "hf_model"))
                # Save action_head weights separately (needed by eval / quantization).
                # clone() detaches each tensor from ZeRO's flattened parameter buffer —
                # torch.save serializes whole storages, so saving the raw views drags
                # the entire 15 GB model buffer into the file.
                torch.save(
                    {k: v.detach().cpu().clone()
                     for k, v in model_engine.module.action_head.state_dict().items()},
                    ckpt_dir / "action_head.pt",
                )
                with open(ckpt_dir / "action_stats.json", "w") as f:
                    json.dump(action_stats, f, indent=2)
                print(f"[Checkpoint] Saved epoch {epoch+1} → {ckpt_dir}")
            except Exception as e:
                print(f"[Checkpoint] WARNING: HF export for epoch {epoch+1} failed "
                      f"({type(e).__name__}: {e}). DeepSpeed checkpoint is still saved.")

            # Prune heavy DeepSpeed optimizer state to keep disk bounded: keep the
            # global_step* dirs for only the last keep_last_n epochs (the small
            # hf_model exports are retained for every epoch for eval). Without this
            # every epoch leaves ~107 GB behind and fills the disk.
            if args.keep_last_n > 0:
                import re, shutil
                cur = epoch + 1
                for ed in Path(args.output_dir).glob("epoch_*"):
                    m = re.match(r"epoch_(\d+)$", ed.name)
                    if not m:
                        continue
                    if int(m.group(1)) <= cur - args.keep_last_n:
                        for gs in ed.glob("global_step*"):
                            if gs.is_dir():
                                shutil.rmtree(gs, ignore_errors=True)
                                print(f"[Prune] removed {gs}")
                        (ed / "latest").unlink(missing_ok=True)
        # Hold all ranks (on the watchdog-free gloo group) until rank 0
        # finishes the ~15 GB HF export above.
        torch.distributed.barrier(group=gloo_group)

        avg_loss = epoch_loss / steps_per_epoch
        if is_main:
            print(f"[Epoch {epoch+1}/{args.num_epochs}] avg_loss={avg_loss:.4f}")
            wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})

        if (epoch + 1) % args.eval_every == 0:
            if is_main:
                try:
                    _run_quick_eval(model_engine, processor, action_tokenizer, action_stats, args, epoch + 1)
                except Exception as e:
                    # A quick-eval failure must not kill a multi-hour training run
                    print(f"[Eval epoch {epoch+1}] quick eval failed, continuing training: {e}")
                    model_engine.module.train()
            # Ranks 1-7 wait here while rank 0 runs rollouts — they must not
            # start the next epoch's NCCL collectives without rank 0.
            torch.distributed.barrier(group=gloo_group)

    if is_main:
        wandb.finish()


def _run_quick_eval(model_engine, processor, action_tokenizer, action_stats, args, epoch):
    """
    3 rollouts × first task of each LIBERO suite — training sanity check only.
    Full evaluation lives in 02_eval_libero.py.
    """
    try:
        from libero.libero import benchmark
        import libero.libero.envs as envs
    except ImportError:
        print("[Eval] LIBERO not importable — skipping. Install with pip install -e LIBERO/")
        return

    model_engine.module.eval()
    results = {}

    action_mean = torch.tensor(action_stats["mean"], device=model_engine.device, dtype=torch.bfloat16)
    action_std  = torch.tensor(action_stats["std"],  device=model_engine.device, dtype=torch.bfloat16)

    for suite_name in LIBERO_SUITES:
        # Installed LIBERO names the long-horizon suite "libero_10", not "libero_long"
        bench_key = "libero_10" if suite_name == "libero_long" else suite_name
        suite     = benchmark.get_benchmark_dict()[bench_key]()
        task      = suite.get_task(0)
        env_args  = suite.get_task_init_states(0)
        env       = envs.OffScreenRenderEnv(**task.get_env_kwargs())

        # Placeholder action token IDs — identical to training input & 02_eval_libero.py
        # (built directly, never round-tripped through text). Used to give the sequence
        # the trailing action-token positions the action_head reads from.
        zero_chunk = np.zeros((OFT_CHUNK_SIZE, OFT_ACTION_DIM), dtype=np.float32)
        disc = np.digitize(np.clip(zero_chunk, -1.0, 1.0), action_tokenizer.bins)
        action_token_ids = torch.from_numpy(
            (processor.tokenizer.vocab_size - disc).reshape(-1).astype(np.int64)
        ).to(model_engine.device)
        action_len = OFT_CHUNK_SIZE * OFT_ACTION_DIM

        sys_msg = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        )

        successes = 0
        for rollout_idx in range(3):
            env.seed(rollout_idx)
            env.reset()
            obs = env.set_init_state(env_args[rollout_idx % len(env_args)])
            instruction = task.language

            # Build the constant input sequence once (prefix IDs + placeholder action IDs,
            # NO EOS at inference → action tokens are the LAST action_len positions).
            prefix = (
                f"{sys_msg}USER: What action should the robot take to {instruction}? "
                f"ASSISTANT: "
            )
            prefix_ids = processor.tokenizer(
                prefix, add_special_tokens=True, return_tensors="pt"
            )["input_ids"].squeeze(0).to(model_engine.device)
            input_ids = torch.cat([prefix_ids, action_token_ids]).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            done = False
            success = False
            step = 0
            while not done and step < 600:
                # Rotate 180° to match the training/eval image convention.
                image = Image.fromarray(
                    np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                )
                pixel_values = processor.image_processor(
                    images=image, return_tensors="pt"
                )["pixel_values"].to(model_engine.device, dtype=torch.bfloat16)

                with torch.no_grad():
                    # Same forward path as training & 02_eval — read action-token hidden
                    # states and map them through OUR action_head (not the base
                    # predict_action, which uses the wrong embedded norm stats).
                    outputs = model_engine.module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        output_hidden_states=True,
                    )
                    action_hidden = outputs.hidden_states[-1][:, -action_len:, :]
                    action_chunk = model_engine.module.action_head.predict_action(action_hidden)

                action_chunk = action_chunk.float().squeeze(0).cpu()          # (chunk, 7) normalized
                action_chunk = action_chunk * action_std.float().cpu() + action_mean.float().cpu()

                for ac_step in range(OFT_CHUNK_SIZE):
                    if step >= 600 or done:
                        break
                    obs, reward, done, info = env.step(action_chunk[ac_step].numpy())
                    step += 1
                    if done:
                        success = True

            if success:
                successes += 1

        env.close()
        results[suite_name] = successes / 3

    model_engine.module.train()
    print(f"[Eval epoch {epoch}] {results}")
    wandb.log({f"eval/{k}_success_rate": v for k, v in results.items()} | {"epoch": epoch})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="openvla/openvla-7b",
                   help="HF model ID. Always openvla/openvla-7b — OFT action_head is created "
                        "separately by this script (not embedded in the HF checkpoint).")
    p.add_argument("--data_root", required=True,
                   help="Root dir containing libero_spatial/, libero_object/, etc.")
    p.add_argument("--output_dir", default="checkpoints/openvla_oft_libero")
    p.add_argument("--run_name",   default="openvla_oft_libero_run1")
    p.add_argument("--ds_config",  default="configs/ds_zero2.json")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--per_gpu_batch_size", type=int, default=8,
                   help="Micro batch per GPU. Safe up to 16 on H100 with ZeRO-2.")
    p.add_argument("--grad_accum", type=int, default=2,
                   help="Gradient accum steps. Effective batch = per_gpu × accum × 8 GPUs.")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps",  type=int, default=200)
    p.add_argument("--eval_every",    type=int, default=2)
    p.add_argument("--log_every",     type=int, default=50)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--max_steps_per_epoch", type=int, default=0,
                   help="If >0, cap batches per epoch. 0 = full epoch (~6.7 h on 8x H100). "
                        "~2500 steps ≈ 30-35 min of training per epoch.")
    p.add_argument("--init_weights", default="",
                   help="Warm-start: load model weights from this checkpoint (raw state "
                        "dict or a DeepSpeed mp_rank_*_model_states.pt) before DeepSpeed "
                        "init, then train with a FRESH optimizer. Use to recover a crashed "
                        "run whose optimizer shards were lost.")
    p.add_argument("--start_epoch", type=int, default=0,
                   help="First epoch index to train. With --init_weights from epoch N, set "
                        "this to N to skip already-completed epochs (saves continue as "
                        "epoch_{start_epoch+1:03d}, ...).")
    p.add_argument("--keep_last_n", type=int, default=1,
                   help="Keep heavy DeepSpeed global_step* optimizer state for only the last "
                        "N epochs (hf_model exports are always kept). 0 = keep everything.")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--local_rank", type=int, default=-1)  # injected by torchrun
    return p.parse_args()


if __name__ == "__main__":
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    args = parse_args()
    train(args)
