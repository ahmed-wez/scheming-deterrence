"""
01_download_model.py

Downloads a model from configs/models.yaml, pins it to an exact commit revision
(so re-running this later — on either Kaggle account — always gets byte-identical
weights), and writes a manifest recording the revision SHA and a sha256 of every
downloaded file. That manifest is your reproducibility record for the paper's
supplementary material.

Usage:
    python scripts/01_download_model.py --target primary
    python scripts/01_download_model.py --target replication

First run for a given target will print a "PIN THIS REVISION" line — copy that
SHA into configs/models.yaml under that model's `revision:` field, then commit
and push, so future runs (on either account) are pinned automatically.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
MANIFEST_DIR = Path(__file__).resolve().parent.parent / "outputs" / "manifests"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main(target: str, confirm_deferred: bool):
    cfg = load_config()
    if target not in cfg["models"]:
        print(f"Unknown target '{target}'. Choices: {list(cfg['models'].keys())}")
        sys.exit(1)

    model_cfg = cfg["models"][target]

    status = model_cfg.get("status", "active")
    if status == "deferred_to_revision" and not confirm_deferred:
        print("=" * 60)
        print(f"'{target}' is marked deferred_to_revision in configs/models.yaml.")
        print("Per docs/plan_amendments.md, this was deferred to save ~200 GPU-hours")
        print("and should not be downloaded yet.")
        print()
        print("If you've deliberately decided to un-defer it (e.g. a reviewer asked")
        print("for the replication), re-run with --confirm-deferred to proceed anyway.")
        print("=" * 60)
        sys.exit(1)

    repo_id = model_cfg["repo_id"]
    pinned_revision = model_cfg.get("revision")
    local_dir = Path(model_cfg["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if model_cfg.get("gated") and not token:
        print(f"{repo_id} is gated and HF_TOKEN is not set. Set it from Kaggle Secrets first.")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"Resolving current revision for {repo_id} ...")
    info = api.model_info(repo_id, revision=pinned_revision or "main", token=token)
    resolved_sha = info.sha
    print(f"  resolved commit SHA: {resolved_sha}")

    if pinned_revision and pinned_revision != resolved_sha:
        print(f"  NOTE: config has a different pinned revision ({pinned_revision}) than 'main' ({resolved_sha}).")
        print("        Proceeding with the pinned revision from config, as intended.")

    download_revision = pinned_revision or resolved_sha

    print(f"Downloading {repo_id} @ {download_revision} -> {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        revision=download_revision,
        local_dir=str(local_dir),
        token=token,
        # Skip files you don't need for a HF transformers load — saves disk/bandwidth.
        ignore_patterns=["*.gguf", "*.onnx", "original/*", "*.pth"],
    )

    print("Computing sha256 manifest of downloaded files (this can take a minute for 16GB)...")
    file_hashes = {}
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(local_dir))
            file_hashes[rel] = sha256_of_file(path)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_id": repo_id,
        "pinned_revision_in_config": pinned_revision,
        "downloaded_revision": download_revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_dir": str(local_dir),
        "file_sha256": file_hashes,
    }
    manifest_path = MANIFEST_DIR / f"{target}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 60)
    print(f"Done. Manifest written to {manifest_path}")
    if not pinned_revision:
        print()
        print(f"  >>> PIN THIS REVISION in configs/models.yaml for '{target}': {resolved_sha}")
        print(f"      models.{target}.revision: {resolved_sha}")
        print("      Then commit + push so both Kaggle accounts use the same pin.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["primary", "replication"])
    parser.add_argument(
        "--confirm-deferred",
        action="store_true",
        help="Required to download a target marked deferred_to_revision in configs/models.yaml",
    )
    args = parser.parse_args()
    main(args.target, args.confirm_deferred)
