"""
02_baseline_smoke_test.py

Loads the downloaded model and runs a handful of generations to confirm the
environment + weights actually work end to end. This is NOT the full safety
benchmark suite — it's a fast sanity check so you don't discover a broken
quantization config 40 minutes into a real eval run.

Tries fp16 first; if that OOMs on a 16GB Kaggle GPU, automatically falls back
to 8-bit (closer to true model behavior than 4-bit, which matters for getting
a clean, citable baseline number).

Usage:
    python scripts/02_baseline_smoke_test.py --target primary
"""

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"

SMOKE_PROMPTS = [
    "What's the capital of France?",
    "Explain what a LoRA adapter is in two sentences.",
    "Write a one-line Python function that reverses a string.",
]


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def try_load(local_dir: str, precision: str):
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    if precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            local_dir, torch_dtype=torch.float16, device_map="auto"
        )
    elif precision == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir, quantization_config=bnb_config, device_map="auto"
        )
    else:
        raise ValueError(f"Unsupported precision for baseline eval: {precision}")
    return tokenizer, model


def main(target: str):
    cfg = load_config()
    model_cfg = cfg["models"][target]
    local_dir = model_cfg["local_dir"]
    preference = cfg["quantization"]["baseline_eval_preference"]

    tokenizer, model, used_precision = None, None, None
    for precision in preference:
        try:
            print(f"Attempting to load in {precision} ...")
            tokenizer, model = try_load(local_dir, precision)
            used_precision = precision
            print(f"  Loaded successfully in {precision}.")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM in {precision}, trying next option in baseline_eval_preference.")
            torch.cuda.empty_cache()
            continue

    if model is None:
        print("Could not load the model in any configured precision. Check GPU/VRAM.")
        return

    print(f"\n--- Running {len(SMOKE_PROMPTS)} smoke-test generations (precision={used_precision}) ---\n")
    for prompt in SMOKE_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            model.device
        )
        output = model.generate(inputs, max_new_tokens=80, do_sample=False)
        text = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        print(f"PROMPT: {prompt}")
        print(f"OUTPUT: {text}\n")

    print("=" * 60)
    print(f"Smoke test passed in {used_precision}. Model + environment are working end to end.")
    print("Next: run the actual safety/capability benchmark suite (e.g. lm-evaluation-harness")
    print("with truthfulqa_mc2 / mmlu / a refusal-consistency set) to get your clean baseline numbers.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["primary", "replication"])
    args = parser.parse_args()
    main(args.target)
