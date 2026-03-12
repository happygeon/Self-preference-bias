#!/usr/bin/env python
"""
prefix_pairwise_attack.py
──────────────────────────
Goal: For each QA answer, find a short prefix that MINIMIZES the judge model's
per‑token perplexity of the whole sequence (BOS + prefix + original answer).
Lower perplexity → higher pair‑wise win‑rate (empirical correlation).

Output: <input_file_basename>_ppl_attack.json
Each record gets:
  - attack_prefix
  - attacked_output  (prefix + original)
  - ppl_before
  - ppl_after
  - ppl_delta
────────────────────────────────────────────────────────────────────
pip install "transformers>=4.52.0" accelerate bitsandbytes torch tqdm
"""

import json, argparse, torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ───── Hyper‑parameters ─────
MAX_LEN      = 10   # prefix 최대 길이
TOPK         = 20   # 각 스텝 후보 수
EPS          = 0.01 # 개선 최소폭 (PPL 차이)

def load_model(model_id: str, device: str):
    """8‑bit 로드 (VRAM 절약)"""
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    mdl = AutoModelForCausalLM.from_pretrained(model_id,
                                               device_map="auto",
                                               quantization_config=bnb)
    tok.pad_token = tok.eos_token
    bos = tok.bos_token_id or tok.eos_token_id
    return tok, mdl.eval(), bos

@torch.no_grad()
def seq_ppl(ids: torch.Tensor, model, pad_id: int) -> float:
    """
    ids: (1, L) LongTensor
    Return: perplexity of ALL tokens (except first) in sequence.
    """
    out   = model(ids[:, :-1])
    logp  = torch.log_softmax(out.logits, dim=-1)
    tgt   = ids[:, 1:]
    nll   = -logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    return nll.mean().exp().item()

def search_prefix(base_text: str,
                  tokenizer,
                  model,
                  bos_id: int,
                  device: str) -> Dict[str, Any]:
    """
    Greedy Top‑K search: add tokens to prefix until no further PPL drop.
    Returns dict with prefix text, ppl_before, ppl_after.
    """
    base_ids = tokenizer.encode(base_text,
                                return_tensors="pt",
                                add_special_tokens=False).to(device)
    prefix_tokens: List[int] = []
    pad_id = tokenizer.pad_token_id

    def get_ppl(pref: List[int]) -> float:
        ids = torch.tensor([[bos_id] + pref], dtype=torch.long, device=device)
        seq = torch.cat([ids, base_ids], dim=1)
        return seq_ppl(seq, model, pad_id)

    ppl_best = get_ppl(prefix_tokens)
    for _ in range(MAX_LEN):
        ctx = torch.tensor([[bos_id] + prefix_tokens],
                           dtype=torch.long, device=device)
        logits = model(ctx).logits[:, -1, :]
        topk   = torch.topk(logits, TOPK, dim=-1).indices.squeeze(0).tolist()

        improved, best_tok, best_ppl = False, None, ppl_best
        for tok in topk:
            p = get_ppl(prefix_tokens + [tok])
            if p + EPS < best_ppl:
                improved, best_tok, best_ppl = True, tok, p
        if not improved:
            break
        prefix_tokens.append(best_tok)
        ppl_best = best_ppl

    prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
    return {
        "attack_prefix": prefix_text,
        "ppl_before": get_ppl([]),
        "ppl_after":  ppl_best
    }

def attack_file(
    input_json: str,
    output_json: str,
    judge_model_id: str,
    target_key: str,
    device: str
):
    tokenizer, model, bos_id = load_model(judge_model_id, device)
    data = json.load(open(input_json, encoding="utf-8"))
    out_records = []

    for rec in tqdm(data, desc="Attacking"):
        base_text = rec[target_key]
        info = search_prefix(base_text, tokenizer, model, bos_id, device)

        attacked_text = info["attack_prefix"] + base_text
        rec_out = rec.copy()
        rec_out.update({
            "attack_prefix" : info["attack_prefix"],
            "attacked_output": attacked_text,
            "ppl_before"    : info["ppl_before"],
            "ppl_after"     : info["ppl_after"],
            "ppl_delta"     : info["ppl_before"] - info["ppl_after"]
        })
        out_records.append(rec_out)

    json.dump(out_records, open(output_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"✅ Saved attacked file → {output_json}")

# ───── CLI ─────
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(
        description="Prefix‑based pair‑wise attack via perplexity minimization")
    parser.add_argument("--input", required=True,
                        help="Path to source JSON (e.g., llama31_alpaca_top100_with_prefix.json)")
    parser.add_argument("--output", default=None,
                        help="Path to save attacked JSON (default: <input>_ppl_attack.json)")
    parser.add_argument("--judge_model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Judge model HF repo")
    parser.add_argument("--field", default="final_output",
                        help="JSON field with answer text to attack")
    args = parser.parse_args()

    out_path = args.output or (Path(args.input).stem + "_ppl_attack.json")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    attack_file(args.input, out_path, args.judge_model, args.field, dev)
