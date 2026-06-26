"""
Benchmark OpenVLA-OFT (AWQ quantized) on NVIDIA Jetson Orin Nano.

Measures and reports:
    - Inference latency (ms per prediction, p50/p90/p99)
    - GPU memory usage (MB, peak)
    - Power draw (watts via tegrastats or jtop)
    - Task success rate (optional, requires LIBERO on Jetson)

Exports all results to CSV for paper tables.

SETUP ON JETSON:
    # Transfer quantized model and this script to Jetson, then:
    scp -r checkpoints/openvla_oft_libero_awq_int8/ jetson:~/models/
    scp openvla_oft_pipeline/04_deploy_benchmark_jetson.py jetson:~/

    # On Jetson (after installing requirements per envs/jetson_requirements.txt):
    python 04_deploy_benchmark_jetson.py \
        --model_dir ~/models/openvla_oft_libero_awq_int8 \
        --action_stats ~/models/openvla_oft_libero_awq_int8/action_stats.json \
        --output_csv ~/results/jetson_benchmark.csv

    # With LIBERO task rollouts (requires LIBERO installed on Jetson):
    python 04_deploy_benchmark_jetson.py \
        --model_dir ~/models/openvla_oft_libero_awq_int8 \
        --action_stats ~/models/openvla_oft_libero_awq_int8/action_stats.json \
        --output_csv ~/results/jetson_benchmark.csv \
        --run_libero_rollouts \
        --data_root ~/libero_datasets \
        --rollouts_per_task 3

ORIN NANO MEMORY BUDGET:
    - Total unified memory: 8 GB (shared CPU + GPU)
    - OS + JetPack baseline: ~2.5 GB
    - Available for model: ~5.5 GB
    - AWQ INT8 (7B params): ~7 GB raw, but fused GEMM gives ~4-5 GB with activation budget
    - AWQ INT4 (7B params): ~3.5 GB — preferred if INT8 OOM
    - If INT8 OOM: use --fuse_layers=False first to diagnose, then switch to INT4

POWER MEASUREMENT:
    tegrastats format (example):
      RAM 3012/7772MB ... VDD_IN 5012mW VDD_CPU_GPU_CV 1250mW VDD_SOC 562mW
    We parse VDD_IN (total board power) and VDD_CPU_GPU_CV (compute subsystem).

    Alternative: jtop from jetson-stats (cleaner API, same data).
    Install: sudo pip3 install jetson-stats && sudo systemctl restart jtop.service
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

OFT_CHUNK_SIZE = 8
OFT_ACTION_DIM = 7
LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
MAX_STEPS_PER_ROLLOUT = 600


# ─────────────────────────────────────────────────────────────────────────────
# Power monitoring

class TegrastatsMonitor:
    """
    Non-blocking tegrastats reader.
    Spawns tegrastats as a background subprocess, parses VDD_IN and VDD_CPU_GPU_CV.
    Call start() before your benchmark, stop() after, then get_stats() for summary.
    """

    # Regex patterns for Orin-family tegrastats output
    _VDD_IN_RE = re.compile(r"VDD_IN\s+(\d+)mW")
    _VDD_CV_RE = re.compile(r"VDD_CPU_GPU_CV\s+(\d+)mW")
    _GPU_FREQ_RE = re.compile(r"GR3D_FREQ\s+(\d+)%@(\d+)")
    _RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._samples: list[dict] = []
        self._proc = None
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self._samples.clear()
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
        except FileNotFoundError:
            print("[Power] tegrastats not found — power monitoring disabled. "
                  "Is this running on a Jetson with JetPack installed?")

    def stop(self):
        self._stop_event.set()
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _read_loop(self):
        for line in self._proc.stdout:
            if self._stop_event.is_set():
                break
            sample = {"timestamp": time.time()}
            m = self._VDD_IN_RE.search(line)
            if m:
                sample["vdd_in_mw"] = int(m.group(1))
            m = self._VDD_CV_RE.search(line)
            if m:
                sample["vdd_cpu_gpu_cv_mw"] = int(m.group(1))
            m = self._RAM_RE.search(line)
            if m:
                sample["ram_used_mb"] = int(m.group(1))
                sample["ram_total_mb"] = int(m.group(2))
            if len(sample) > 1:  # has at least one real measurement
                self._samples.append(sample)

    def get_stats(self) -> dict:
        if not self._samples:
            return {"error": "No samples collected — tegrastats not available"}

        vdd_in = [s["vdd_in_mw"] for s in self._samples if "vdd_in_mw" in s]
        vdd_cv = [s["vdd_cpu_gpu_cv_mw"] for s in self._samples if "vdd_cpu_gpu_cv_mw" in s]
        ram = [s["ram_used_mb"] for s in self._samples if "ram_used_mb" in s]

        stats = {"num_samples": len(self._samples)}
        if vdd_in:
            stats.update({
                "vdd_in_mean_w": np.mean(vdd_in) / 1000,
                "vdd_in_peak_w": np.max(vdd_in) / 1000,
            })
        if vdd_cv:
            stats.update({
                "vdd_cpu_gpu_cv_mean_w": np.mean(vdd_cv) / 1000,
                "vdd_cpu_gpu_cv_peak_w": np.max(vdd_cv) / 1000,
            })
        if ram:
            stats.update({
                "ram_mean_mb": np.mean(ram),
                "ram_peak_mb": np.max(ram),
            })
        return stats


class JtopMonitor:
    """
    Alternative power monitor using jtop (jetson-stats).
    Cleaner API than tegrastats. Requires: sudo pip3 install jetson-stats
    """

    def __init__(self):
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        try:
            from jtop import jtop
            self._jtop = jtop()
            self._jtop.start()
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
        except ImportError:
            print("[Power] jtop not found. Install: sudo pip3 install jetson-stats")
        except Exception as e:
            print(f"[Power] jtop start failed: {e}")

    def _read_loop(self):
        while not self._stop_event.is_set():
            try:
                stats = self._jtop.stats
                self._samples.append({
                    "timestamp": time.time(),
                    "power_total_mw": stats.get("Power TOT", 0),
                    "gpu_util_pct": stats.get("GPU", 0),
                    "ram_used_mb": stats.get("RAM", {}).get("use", 0),
                })
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        try:
            self._jtop.close()
        except Exception:
            pass

    def get_stats(self) -> dict:
        if not self._samples:
            return {"error": "No jtop samples"}
        power = [s["power_total_mw"] for s in self._samples]
        ram = [s["ram_used_mb"] for s in self._samples]
        return {
            "num_samples": len(self._samples),
            "power_total_mean_w": np.mean(power) / 1000,
            "power_total_peak_w": np.max(power) / 1000,
            "ram_mean_mb": np.mean(ram),
            "ram_peak_mb": np.max(ram),
        }


def get_power_monitor(prefer_jtop: bool = False):
    if prefer_jtop:
        try:
            import jtop  # noqa: F401
            return JtopMonitor()
        except ImportError:
            pass
    return TegrastatsMonitor()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (AWQ)

def load_awq_model(model_dir: str, device: torch.device):
    """
    Load AWQ-quantized model for inference.

    fuse_layers=True enables fused GEMM kernels — ~2x faster on Jetson GPU.
    Set fuse_layers=False if you see CUDA errors; it's a safe fallback.
    """
    from awq import AutoAWQForCausalLM

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    print(f"[Deploy] Loading AWQ model from {model_dir}...")
    model = AutoAWQForCausalLM.from_quantized(
        model_dir,
        fuse_layers=True,
        trust_remote_code=True,
        # Do NOT use device_map="auto" on Jetson — Orin uses unified memory
        # and device_map will add unnecessary CPU↔GPU copies
    )

    # Move to Jetson GPU
    model = model.to(device).eval()

    # Verify action head survived quantization
    assert hasattr(model, "action_head"), (
        "action_head missing from quantized model. "
        "The quantization script may not have correctly re-attached the fp16 action head. "
        "Re-run 03_quantize_awq.py and check the 'Re-attaching action head' log line."
    )

    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# Latency benchmark

def benchmark_inference_latency(
    model,
    processor,
    device: torch.device,
    num_warmup: int = 10,
    num_benchmark: int = 100,
) -> dict:
    """
    Measure pure inference latency (no sim, no env step).
    Uses a fixed dummy image and instruction to isolate model latency.
    Warmup runs are excluded from statistics.
    """
    print(f"[Benchmark] Measuring inference latency ({num_warmup} warmup + {num_benchmark} timed)...")

    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    dummy_instruction = "pick up the red block and place it on the blue plate"
    dummy_proprio = torch.zeros(1, 9, device=device, dtype=torch.bfloat16)

    inputs = processor(text=dummy_instruction, images=dummy_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # Warmup — fills CUDA caches, JIT compiles fused kernels
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.predict_action(**inputs, proprio=dummy_proprio)
        torch.cuda.synchronize()

    # Timed runs
    latencies_ms = []
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(num_benchmark):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.predict_action(**inputs, proprio=dummy_proprio)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

    latencies = np.array(latencies_ms)
    return {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p90_ms": float(np.percentile(latencies, 90)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_std_ms": float(np.std(latencies)),
        "peak_gpu_mem_mb": peak_mem_mb,
        "num_runs": num_benchmark,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LIBERO rollout eval on Jetson (optional)

def run_libero_eval_on_jetson(
    model,
    processor,
    action_stats: dict,
    data_root: str,
    device: torch.device,
    rollouts_per_task: int = 3,
    max_tasks_per_suite: int | None = None,
) -> list[dict]:
    """
    Run LIBERO rollouts on Jetson to measure task success rate at deployment.
    This is intentionally limited (rollouts_per_task=3) because sim is slow on Jetson.
    """
    try:
        from libero.libero import benchmark
        import libero.libero.envs as libero_envs
    except ImportError:
        print("[Eval] LIBERO not installed on Jetson — skipping rollout eval.")
        return []

    action_mean = torch.tensor(action_stats["mean"], device=device, dtype=torch.bfloat16)
    action_std = torch.tensor(action_stats["std"], device=device, dtype=torch.bfloat16)

    all_results = []

    for suite_name in LIBERO_SUITES:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task_names = suite.get_task_names()
        if max_tasks_per_suite:
            task_names = task_names[:max_tasks_per_suite]

        for task_idx, task_name in enumerate(task_names):
            task = suite.get_task(task_idx)
            init_states = suite.get_task_init_states(task_idx)
            env = libero_envs.OffScreenRenderEnv(**task.get_env_kwargs())

            for rollout_idx in range(rollouts_per_task):
                env.seed(rollout_idx)
                obs, _ = env.reset()
                env.set_init_state(init_states[rollout_idx % len(init_states)])

                done = False
                step = 0
                latencies = []
                t_rollout = time.time()

                while not done and step < MAX_STEPS_PER_ROLLOUT:
                    image = Image.fromarray(obs["agentview_rgb"])
                    eef_pos = torch.tensor(obs["robot0_eef_pos"], dtype=torch.float32)
                    eef_quat = torch.tensor(obs["robot0_eef_quat"], dtype=torch.float32)
                    gripper = torch.tensor(obs["robot0_gripper_qpos"], dtype=torch.float32)
                    proprio = torch.cat([eef_pos, eef_quat, gripper]).unsqueeze(0).to(
                        device, dtype=torch.bfloat16
                    )

                    inputs = processor(text=task.language, images=image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                    t_inf = time.perf_counter()
                    with torch.no_grad():
                        action_chunk = model.predict_action(**inputs, proprio=proprio)
                    torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - t_inf) * 1000)

                    action_chunk = action_chunk.squeeze(0) * action_std + action_mean
                    for ac_step in range(OFT_CHUNK_SIZE):
                        if step >= MAX_STEPS_PER_ROLLOUT or done:
                            break
                        action_np = action_chunk[ac_step].float().cpu().numpy()
                        obs, _, done, _, info = env.step(action_np)
                        step += 1

                all_results.append({
                    "suite": suite_name,
                    "task": task_name,
                    "rollout_idx": rollout_idx,
                    "success": bool(info.get("success", False)),
                    "steps": step,
                    "rollout_elapsed_sec": time.time() - t_rollout,
                    "inference_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
                    "inference_p90_ms": float(np.percentile(latencies, 90)) if latencies else 0.0,
                })

            env.close()

        suite_results = [r for r in all_results if r["suite"] == suite_name]
        sr = sum(r["success"] for r in suite_results) / max(len(suite_results), 1)
        print(f"[Jetson Eval] {suite_name}: {sr:.1%}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark entry point

def run_full_benchmark(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[Warning] No CUDA device — benchmark will not reflect Jetson GPU performance.")

    with open(args.action_stats) as f:
        action_stats = json.load(f)

    # Load model
    processor, model = load_awq_model(args.model_dir, device)

    # Record baseline memory after model load
    if device.type == "cuda":
        model_mem_mb = torch.cuda.memory_allocated(device) / 1e6
        print(f"[Deploy] Model loaded. GPU memory: {model_mem_mb:.0f} MB")
    else:
        model_mem_mb = 0.0

    # Start power monitoring
    monitor = get_power_monitor(prefer_jtop=args.use_jtop)
    monitor.start()
    time.sleep(1.0)  # let monitor stabilize before benchmark

    # Latency benchmark
    latency_stats = benchmark_inference_latency(
        model, processor, device,
        num_warmup=args.warmup_runs,
        num_benchmark=args.benchmark_runs,
    )

    print("\n[Deploy] Latency Results:")
    for k, v in latency_stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # LIBERO rollout eval (optional)
    rollout_results = []
    if args.run_libero_rollouts:
        print("\n[Deploy] Running LIBERO rollouts on Jetson...")
        rollout_results = run_libero_eval_on_jetson(
            model=model,
            processor=processor,
            action_stats=action_stats,
            data_root=args.data_root,
            device=device,
            rollouts_per_task=args.rollouts_per_task,
            max_tasks_per_suite=args.max_tasks_per_suite,
        )

    # Stop power monitor and collect stats
    monitor.stop()
    power_stats = monitor.get_stats()

    print("\n[Deploy] Power Results:")
    for k, v in power_stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Export CSV ────────────────────────────────────────────────────────────
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Summary row (for paper table)
    summary = {
        "model_dir": str(args.model_dir),
        "device": str(device),
        "model_mem_mb": model_mem_mb,
        **latency_stats,
        **{f"power_{k}": v for k, v in power_stats.items() if k != "error"},
    }
    if rollout_results:
        suite_srs = {}
        for suite in LIBERO_SUITES:
            suite_r = [r for r in rollout_results if r["suite"] == suite]
            if suite_r:
                suite_srs[f"success_{suite}"] = sum(r["success"] for r in suite_r) / len(suite_r)
        summary.update(suite_srs)
        if len(suite_srs) == 4:
            summary["success_avg"] = sum(suite_srs.values()) / 4

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # Detailed rollout CSV (separate file)
    if rollout_results:
        detailed_path = output_path.with_stem(output_path.stem + "_rollouts")
        with open(detailed_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rollout_results[0].keys()))
            writer.writeheader()
            writer.writerows(rollout_results)
        print(f"[Deploy] Detailed rollout results: {detailed_path}")

    # JSON summary for programmatic use
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Deploy] Results saved to {output_path}")
    print(f"[Deploy] JSON summary: {json_path}")

    # ── Paper-table-friendly print ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PAPER TABLE ROW")
    print("=" * 60)
    print(f"  Latency (mean):    {latency_stats['latency_mean_ms']:.1f} ms")
    print(f"  Latency (p90):     {latency_stats['latency_p90_ms']:.1f} ms")
    print(f"  GPU Memory (peak): {latency_stats['peak_gpu_mem_mb']:.0f} MB")
    print(f"  Model footprint:   {model_mem_mb:.0f} MB")
    if "vdd_in_mean_w" in power_stats:
        print(f"  Power (mean):      {power_stats['vdd_in_mean_w']:.2f} W")
        print(f"  Power (peak):      {power_stats['vdd_in_peak_w']:.2f} W")
    if "success_avg" in summary:
        print(f"  LIBERO Avg SR:     {summary['success_avg']:.1%}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="Path to AWQ-quantized model directory")
    p.add_argument("--action_stats", required=True,
                   help="Path to action_stats.json")
    p.add_argument("--output_csv", default="results/jetson_benchmark.csv")

    # Latency benchmark options
    p.add_argument("--warmup_runs", type=int, default=10)
    p.add_argument("--benchmark_runs", type=int, default=100)

    # Power monitor
    p.add_argument("--use_jtop", action="store_true",
                   help="Use jtop instead of tegrastats for power monitoring")

    # LIBERO rollout eval (optional)
    p.add_argument("--run_libero_rollouts", action="store_true")
    p.add_argument("--data_root", default=None,
                   help="LIBERO dataset root (required if --run_libero_rollouts)")
    p.add_argument("--rollouts_per_task", type=int, default=3)
    p.add_argument("--max_tasks_per_suite", type=int, default=None,
                   help="Limit tasks per suite (default: all 10)")
    return p.parse_args()


if __name__ == "__main__":
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    args = parse_args()
    if args.run_libero_rollouts and args.data_root is None:
        print("ERROR: --data_root is required when --run_libero_rollouts is set.")
        sys.exit(1)
    run_full_benchmark(args)
