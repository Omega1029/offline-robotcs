"""
Pre-flight checks before running any pipeline step.

Run this after setup_notes.sh and before launch_train.sh to catch
device placement errors, version mismatches, and data path issues early.

USAGE:
    python verify_setup.py --data_root /path/to/libero_datasets
    python verify_setup.py --data_root /path/to/libero_datasets --check_model
    python verify_setup.py --skip_model --skip_libero   # CI / quick check
"""

import argparse
import json
import os
import sys
from pathlib import Path

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "      "

failures = []
warnings = []


def check(label, condition, fail_msg="", warn=False):
    if condition:
        print(f"{PASS} {label}")
    elif warn:
        warnings.append(label)
        print(f"{WARN} {label}" + (f" — {fail_msg}" if fail_msg else ""))
    else:
        failures.append(label)
        print(f"{FAIL} {label}" + (f" — {fail_msg}" if fail_msg else ""))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Python / CUDA / GPU

def check_environment():
    print("\n── Environment ──────────────────────────────────────────────")

    import platform
    check("Python 3.10.x", sys.version_info[:2] == (3, 10),
          f"got {sys.version_info[:2]} — training env requires 3.10")

    import torch
    check("CUDA available", torch.cuda.is_available(),
          "torch.cuda.is_available() returned False — check driver and torch wheel")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        check(f"GPU count ≥ 8 (found {n})", n >= 8,
              "Training script expects 8 GPUs. Adjust --nproc-per-node if fewer.",
              warn=(n < 8))
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"{INFO}  GPU {i}: {name} ({mem_gb:.0f} GB)")

    # CUDA version from torch vs driver
    cuda_ver = torch.version.cuda
    print(f"{INFO}  torch.version.cuda = {cuda_ver}")
    check("CUDA 12.x (H100 node)", cuda_ver and cuda_ver.startswith("12"),
          f"Got CUDA {cuda_ver}. H100 requires CUDA ≥ 12.0. Check torch wheel.",
          warn=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Package versions

REQUIRED_VERSIONS = {
    "transformers": ("4.40.2", "4.40."),
    "accelerate":   ("0.28.0", "0.28."),
    "peft":         ("0.10.0", "0.10."),
    "deepspeed":    ("0.14.2", "0.14."),
    "numpy":        ("1.26.4", "1."),
    "robosuite":    ("1.4.1",  "1.4."),
    "tokenizers":   ("0.19.1", "0.19."),
}


def check_versions():
    print("\n── Package Versions ─────────────────────────────────────────")
    import importlib

    for pkg, (exact, prefix) in REQUIRED_VERSIONS.items():
        try:
            mod = importlib.import_module(pkg)
            ver = mod.__version__
            ok = ver.startswith(prefix)
            is_transformers_critical = (pkg == "transformers" and ver != exact)
            if is_transformers_critical and not ok:
                check(f"{pkg}=={ver}", False,
                      f"CRITICAL: must be exactly {exact}. "
                      f"transformers≥4.45 silently breaks OFT action head.")
            else:
                check(f"{pkg}=={ver}", ok,
                      f"expected ~{exact}", warn=(pkg != "transformers"))
        except ImportError:
            check(f"{pkg} installed", False, "not found — run: pip install -r envs/training_env.yml")

    # autoawq (only needed for quantization step)
    try:
        from awq import AutoAWQForCausalLM  # noqa
        import awq
        check(f"autoawq=={awq.__version__}", True)
    except ImportError:
        check("autoawq installed", False, "needed for step 3. pip install autoawq==0.2.5", warn=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIBERO data paths

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]


def check_libero_data(data_root: str):
    print("\n── LIBERO Datasets ──────────────────────────────────────────")
    root = Path(data_root)
    check(f"DATA_ROOT exists: {data_root}", root.exists(),
          "set --data_root to the directory containing libero_spatial/ etc.")

    if not root.exists():
        return

    for suite in LIBERO_SUITES:
        suite_dir = root / suite
        hdf5_files = list(suite_dir.glob("*.hdf5"))
        check(f"{suite}: {len(hdf5_files)} HDF5 files", len(hdf5_files) > 0,
              f"{suite_dir} is empty or missing. Run the LIBERO download in setup_notes.sh.")

        if hdf5_files:
            # Spot-check one file to verify HDF5 structure
            import h5py
            try:
                with h5py.File(hdf5_files[0], "r") as f:
                    has_data = "data" in f
                    first_demo = list(f["data"].keys())[0] if has_data else None
                    has_actions = has_data and "actions" in f["data"][first_demo]
                    has_rgb = has_data and "agentview_rgb" in f["data"][first_demo]["obs"]
                check(f"  {suite} HDF5 structure valid", has_actions and has_rgb,
                      "Missing expected keys. Dataset may be corrupted.")
            except Exception as e:
                check(f"  {suite} HDF5 readable", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 4. MUJOCO / rendering

def check_mujoco():
    print("\n── MuJoCo / Rendering ───────────────────────────────────────")
    gl = os.environ.get("MUJOCO_GL", "")
    check("MUJOCO_GL=egl", gl == "egl",
          f"Got MUJOCO_GL='{gl}'. Set to 'egl' for headless server rendering. "
          f"'osmesa' works but is ~3x slower.",
          warn=(gl == "osmesa"))

    try:
        import mujoco
        check(f"mujoco=={mujoco.__version__}", True)
    except ImportError:
        check("mujoco installed", False, "pip install mujoco==2.3.7")

    try:
        import mujoco
        # Quick render test
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        d = mujoco.MjData(m)
        mujoco.mj_step(m, d)
        check("MuJoCo step() works", True)
    except Exception as e:
        check("MuJoCo step() works", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 5. openvla-oft model and action head

def check_model():
    print("\n── OpenVLA-OFT Model ─────────────────────────────────────────")
    print(f"{INFO}  Loading moojink/openvla-oft... (cached after first download)")

    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            "moojink/openvla-oft", trust_remote_code=True
        )
        check("Processor loaded (trust_remote_code)", True)

        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        check("Model loaded with AutoModelForVision2Seq", True)
        check("model.config.model_type == 'openvla'",
              model.config.model_type == "openvla",
              f"Got '{model.config.model_type}'. Wrong model or wrong class.")
        check("model has action_head",
              hasattr(model, "action_head"),
              "action_head missing. transformers version mismatch — pin to 4.40.2.")

        if hasattr(model, "action_head"):
            print(f"{INFO}  action_head: {model.action_head}")

        # Find inference method
        import inspect
        predict_methods = [
            m for m in dir(model)
            if ("predict" in m.lower() or "action" in m.lower())
            and not m.startswith("_")
            and callable(getattr(model, m))
        ]
        check("predict_action method exists",
              "predict_action" in predict_methods,
              f"Method not found. Available: {predict_methods}\n"
              f"     Update every 'model.predict_action(...)' call in the pipeline scripts.")
        print(f"{INFO}  Action-related methods: {predict_methods}")

        # Check forward() signature for proprio param
        sig = inspect.signature(model.forward)
        params = list(sig.parameters.keys())
        proprio_params = [p for p in params if "proprio" in p.lower() or "state" in p.lower()]
        check("forward() has proprio-like param",
              len(proprio_params) > 0,
              f"No proprio param in forward({params}). "
              f"Update 'proprio=' kwarg in training/eval scripts.",
              warn=(len(proprio_params) == 0))
        if proprio_params:
            print(f"{INFO}  Proprio param name: {proprio_params[0]}")
            if proprio_params[0] != "proprio":
                print(f"{WARN} Proprio param is '{proprio_params[0]}', not 'proprio'.")
                print(f"{INFO}  Update all 'proprio=proprio' kwargs in the pipeline scripts.")

    except Exception as e:
        check("Model load succeeded", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 6. DeepSpeed

def check_deepspeed():
    print("\n── DeepSpeed ────────────────────────────────────────────────")
    try:
        import deepspeed
        check(f"deepspeed=={deepspeed.__version__}", True)

        result = deepspeed.ops.op_builder.FusedAdamBuilder().is_compatible()
        check("FusedAdam available", result,
              "FusedAdam not compiled. Run: DS_BUILD_FUSED_ADAM=1 pip install deepspeed==0.14.2 --no-binary deepspeed",
              warn=True)
    except Exception as e:
        check("deepspeed available", False, str(e))

    # Check ds_zero2.json is valid JSON
    config_path = Path("configs/ds_zero2.json")
    if config_path.exists():
        try:
            with open(config_path) as f:
                json.load(f)
            check("configs/ds_zero2.json is valid JSON", True)
        except json.JSONDecodeError as e:
            check("configs/ds_zero2.json is valid JSON", False, str(e))
    else:
        check("configs/ds_zero2.json exists", False,
              "run this script from the openvla_oft_pipeline/ directory")


# ─────────────────────────────────────────────────────────────────────────────
# Summary

def print_summary():
    print("\n" + "=" * 60)
    if not failures and not warnings:
        print(f"{PASS} All checks passed. Ready to train.")
        print("  Next: bash launch_train.sh")
    elif not failures:
        print(f"{WARN} Checks passed with warnings ({len(warnings)} warnings).")
        print("  Warnings above are non-blocking but should be reviewed.")
        for w in warnings:
            print(f"  - {w}")
    else:
        print(f"{FAIL} {len(failures)} check(s) failed — fix before running training.")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=None,
                   help="Root dir with libero_spatial/, libero_object/, etc.")
    p.add_argument("--check_model", action="store_true",
                   help="Load openvla-oft from HuggingFace and verify action head (downloads ~14GB first time)")
    p.add_argument("--skip_libero", action="store_true")
    p.add_argument("--skip_model", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("OpenVLA-OFT Pipeline — Pre-flight Checks")

    check_environment()
    check_versions()
    check_mujoco()
    check_deepspeed()

    if not args.skip_libero:
        if args.data_root:
            check_libero_data(args.data_root)
        else:
            print(f"\n{WARN} Skipping LIBERO data check — pass --data_root to enable.")

    if args.check_model and not args.skip_model:
        check_model()
    elif not args.skip_model:
        print(f"\n{INFO}Skipping model download check (pass --check_model to enable).")

    print_summary()
