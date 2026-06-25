# Scheming Deterrence Experiment — Setup Repo

> **Read this first:** `docs/plan_amendments.md` is the canonical, living record
> of the compute-reduction decision. Seeds, interaction counts, decay-test
> scope, and the Qwen replication's deferred status all live in
> `configs/study_design.yaml` and `configs/models.yaml` now — not in the
> original plan document. If a phase's instructions ever conflict with those
> two files, the files win.

This repo is the shared codebase for the project, designed to be cloned identically
into Kaggle notebooks on **two different Kaggle accounts**, so both get the same
code and you're not copy-pasting between them.

## One-time account setup (do this once, not per-session)

### 1. Hugging Face
1. Create an account at https://huggingface.co (if you don't have one).
2. Go to https://huggingface.co/settings/tokens → create a **Read** token. Save it somewhere safe.
3. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct → click "Agree and access repository".
   This is a gated model — Meta requires you to accept their license. Approval is
   usually instant or within a few hours.
4. Qwen 2.5 7B Instruct (https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) is **not** gated —
   nothing to do there.

### 2. GitHub
1. Create a repo (e.g. `scheming-deterrence`), private is fine.
2. Settings → Developer settings → Personal access tokens → generate a **fine-grained
   token** scoped to just this repo, with read/write contents permission.
3. Push this scaffold to it as your starting commit.

### 3. Kaggle (repeat this part on BOTH accounts)
1. Account → Settings → Phone Verification → verify your phone number.
   (Required by Kaggle before it will give you GPU + internet access on notebooks.)
2. Create a new Notebook. In the right-hand panel:
   - **Accelerator** → GPU T4 x2 (prefer this over P100 — better fp16 throughput).
   - **Internet** → On.
3. Add-ons → Secrets → add two secrets:
   - `HF_TOKEN` = your Hugging Face token
   - `GH_TOKEN` = your GitHub fine-grained token

Do this on **both** Kaggle accounts so either one can clone/pull/push identically.

## Every-session workflow (top of every Kaggle notebook)

```python
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
os.environ["GH_TOKEN"] = secrets.get_secret("GH_TOKEN")
```

```bash
!git clone https://$GH_TOKEN@github.com/<your-username>/scheming-deterrence.git repo
%cd repo
!pip install -q -r requirements.txt
```

If the repo is already cloned from a previous session and you just want the latest code:

```bash
%cd repo
!git pull
```

When you've changed code and want the *other* account to see it next time it runs:

```bash
!git config user.email "you@example.com"
!git config user.name "Your Name"
!git add -A
!git commit -m "describe what changed"
!git push
```

**Important — keep large files out of git entirely.** Model weights, checkpoints, and
datasets should never be committed (GitHub will reject anything close to 100MB anyway).
Two places to put them instead:
- **Kaggle Datasets**: after downloading a model once, save the notebook's output as a
  private Kaggle Dataset, then attach it as an input in future sessions/accounts instead
  of re-downloading. This is the single biggest time-saver given your 30hr/week limit —
  re-downloading a 16GB model every session burns hours and bandwidth for nothing.
- **Your own private HF Hub repo**: good for LoRA adapters later (they're small, MBs not GBs).

`.gitignore` in this scaffold already excludes weight files, checkpoints, and notebook
output so you don't do this by accident.

## What's in this scaffold

```
scripts/00_verify_environment.py   — sanity-check GPU/torch/bitsandbytes before anything else
scripts/01_download_model.py       — downloads + pins + hash-records a model
scripts/02_baseline_smoke_test.py  — loads the model and runs a quick generation check
configs/models.yaml                — the two model IDs, revision pins, quantization defaults
```

## Run order (this step of the plan)

```bash
!python scripts/00_verify_environment.py
!python scripts/01_download_model.py --target primary
!python scripts/02_baseline_smoke_test.py --target primary
```

Note: `--target replication` (Qwen 2.5 7B) is currently deferred to the
revision stage (see `docs/plan_amendments.md`). The script will refuse to run
it without an explicit `--confirm-deferred` flag, so don't bother with it yet
— only download/benchmark the primary model for now.

`01_download_model.py` will print a pinned commit SHA the first time you run it — copy
that into `configs/models.yaml` under `revision:` for that model, commit, and push. From
then on, every future download (on either account) is pinned to that exact snapshot, so
nobody accidentally gets a different version of the model later if Meta/Qwen update the repo.
