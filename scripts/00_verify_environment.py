"""
00_verify_environment.py

Run this FIRST in every fresh Kaggle session, before downloading anything.
It checks that the GPU, CUDA, torch, and bitsandbytes are all in a working state,
so you find out about a broken environment in 10 seconds instead of after a
15-minute model download.

Usage:
    python scripts/00_verify_environment.py
"""

import sys
import subprocess


def check_torch_cuda():
    import torch

    print(f"torch version:       {torch.__version__}")
    print(f"CUDA available:      {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  -> No GPU visible. In Kaggle: Settings panel -> Accelerator -> GPU T4 x2.")
        return False

    n_gpus = torch.cuda.device_count()
    print(f"GPU count:            {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name}  ({vram_gb:.1f} GB VRAM)")

        if "P100" in props.name:
            print("    Note: P100 (Pascal) — fp16 tensor core throughput is weaker than T4.")
        if vram_gb < 15:
            print("    Warning: less than ~15GB VRAM. An 8B model in fp16 will be very tight.")
    return True


def check_bitsandbytes():
    try:
        import bitsandbytes as bnb

        print(f"bitsandbytes version: {bnb.__version__}")
        return True
    except Exception as e:
        print(f"bitsandbytes import FAILED: {e}")
        print("  -> pip install -U bitsandbytes and restart the kernel.")
        return False


def check_transformers_and_hub():
    import transformers
    import huggingface_hub

    print(f"transformers version: {transformers.__version__}")
    print(f"huggingface_hub vers: {huggingface_hub.__version__}")
    return True


def check_hf_token():
    import os

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var is NOT set.")
        print("  -> In Kaggle: load it from Secrets at the top of your notebook, e.g.")
        print('     os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")')
        return False
    print("HF_TOKEN is set (length only, not printing the value).")
    print(f"  length: {len(token)} chars")
    return True


def check_disk_space():
    try:
        out = subprocess.run(["df", "-h", "."], capture_output=True, text=True).stdout
        print("Disk space (current dir):")
        print(out)
    except Exception as e:
        print(f"Could not check disk space: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    results = {
        "torch_cuda": check_torch_cuda(),
        "bitsandbytes": check_bitsandbytes(),
        "transformers_hub": check_transformers_and_hub(),
        "hf_token": check_hf_token(),
    }
    check_disk_space()

    print("=" * 60)
    if all(results.values()):
        print("All checks passed. Safe to proceed to 01_download_model.py")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"FAILED checks: {failed}")
        print("Fix these before downloading a 16GB model — debugging is much faster now.")
        sys.exit(1)
