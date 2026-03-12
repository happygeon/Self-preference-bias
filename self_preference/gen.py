"""
Script to generate QA‑set responses for Qwen3‑8B and Meta‑Llama‑3‑8B locally with pure PyTorch batch generation and tqdm progress.

Fixes in this revision
----------------------
* **Removed all sampling kwargs** → no more “temperature/top_p/top_k invalid” warning.
* **Left‑padding** enforced (`tokenizer.padding_side='left'`) to silence decoder‑only right‑padding warning.
* Adds pad token if missing (`tokenizer.pad_token = eos_token`).

Requirements
------------
```bash
pip install transformers accelerate bitsandbytes huggingface_hub tqdm torch
```
"""
import json
from math import ceil
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

BATCH_SIZE = 8  # tune per VRAM
MAX_NEW_TOKENS = 2048
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################################
# Model & data loaders
################################################################################

def load_model(model_id: str):
    """Return (tokenizer, model) in 8‑bit on single GPU."""
    qconfig = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Ensure left‑padding for decoder‑only models
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=qconfig,
        device_map={"": DEVICE.index if DEVICE.type == "cuda" else DEVICE},
    )
    model.eval()
    return tokenizer, model

def load_eval_set() -> List[dict]:
    repo_id = "tatsu-lab/alpaca_eval"
    filename = "alpaca_eval.json"
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

################################################################################
# Generation helpers
################################################################################

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def generate_responses(tokenizer, model, eval_set: List[dict]):
    """Batch‑generate responses with greedy decoding (no sampling flags)."""

    outputs = []
    total_batches = ceil(len(eval_set) / BATCH_SIZE)

    for batch in tqdm(chunk(eval_set, BATCH_SIZE), total=total_batches, desc="Batches"):
        prompts = [ex.get("instruction") or ex.get("question") for ex in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                do_sample=False,           # greedy decoding, no sampling params
                max_new_tokens=MAX_NEW_TOKENS,
            )
        answers = tokenizer.batch_decode(
            gen_ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for prompt, ans in zip(prompts, answers):
            outputs.append({"instruction": prompt, "output": ans.strip()})
    return outputs

################################################################################
# Main
################################################################################

def main():
    print("Downloading QA set …")
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} instructions.")

    models = {
        "Llama31-8B": "meta-llama/Llama-3.1-8B-Instruct",
    }

    for name, repo in models.items():
        print(f"\n=== {name} ===")
        tokenizer, model = load_model(repo)
        responses = generate_responses(tokenizer, model, eval_set)

        fname = f"{name.replace('/', '_')}_outputs.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
        print(f"Saved → {fname}")

if __name__ == "__main__":
    main()
