#!/usr/bin/env python
"""
alpaca_suffix_search_log.py
───────────────────────────
Llama‑3.1‑8B‑Instruct × Alpaca‑Eval Top‑100
기존 JSON의 base_output 재활용 → suffix 탐색 + 스텝별 로그
"""

import json, math, torch, datetime
from pathlib import Path
from typing import List, Dict, Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ───── 경로 및 파라미터 ─────
HF_MODEL_ID  = "meta-llama/Llama-3.1-8B-Instruct"
BASE_JSON    = "llama31_alpaca_top100_with_prefix.json"   # ← base_output 보관
OUT_PREFIX   = "llama31_alpaca_top100"
LOG_PATH     = Path(f"{OUT_PREFIX}_suffix_search.log.jsonl")
RESULT_JSON  = Path(f"{OUT_PREFIX}_with_suffix.json")

MAX_LEN, TOPK, EPS = 10, 20, 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───── 모델 로드(4‑bit, 단일 GPU) ─────
quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=False)
model     = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, quantization_config=quant_cfg).to(DEVICE)
model.eval()
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ───── base_output 읽기 ─────
data: List[Dict[str, Any]] = json.loads(Path(BASE_JSON).read_text())
texts = [ex["base_output"] for ex in data]
print(f"▶ base_output {len(texts)}개 불러옴")

# ───── PPL 함수 ─────
@torch.no_grad()
def mean_ppl(txts: List[str], batch: int = 4) -> float:
    loss_sum = tok_sum = 0
    for i in range(0, len(txts), batch):
        toks = tokenizer(txts[i:i+batch], return_tensors="pt", padding=True).to(DEVICE)
        loss  = model(**toks, labels=toks.input_ids).loss
        mask  = toks.input_ids.ne(tokenizer.pad_token_id)
        loss_sum += loss.item() * mask.sum().item()
        tok_sum  += mask.sum().item()
    return math.exp(loss_sum / tok_sum)

# ───── suffix 탐색 + 로그 ─────
LOG_PATH.unlink(missing_ok=True)
def log(step, tid, suf_ids, ppl):
    rec = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "step": step,
        "token_id": tid,
        "token_str": tokenizer.decode([tid]) if tid is not None else None,
        "suffix": tokenizer.decode(suf_ids),
        "ppl": round(ppl, 4),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

suffix_ids: List[int] = []
cur_ppl = mean_ppl(texts)
log(0, None, suffix_ids, cur_ppl)
print(f"▸ 시작 PPL: {cur_ppl:.2f}")

for step in range(1, MAX_LEN + 1):
    # 후보 토큰: 각 응답+현재 suffix 의 마지막 logits 합산 상위 TOPK
    logits_sum = torch.zeros(model.config.vocab_size, device=DEVICE)
    for t in texts[:64]:                                          # 속도용 샘플 64개
        ids = tokenizer(t + tokenizer.decode(suffix_ids), return_tensors="pt").input_ids.to(DEVICE)
        logits_sum += model(ids).logits[0, -1]
    cand_ids = logits_sum.topk(TOPK).indices.tolist()

    best_tok = best_ppl = None
    for tid in cand_ids:
        suf = tokenizer.decode(suffix_ids + [tid])
        ppl = mean_ppl([txt + suf for txt in texts])
        if best_ppl is None or ppl < best_ppl:
            best_tok, best_ppl = tid, ppl

    if cur_ppl - best_ppl < EPS:
        print("  ↳ 개선폭 < EPS, 탐색 종료")
        break

    suffix_ids.append(best_tok)
    cur_ppl = best_ppl
    log(step, best_tok, suffix_ids, cur_ppl)
    print(f"  step {step}: +{tokenizer.decode([best_tok]).replace(chr(10),'↵')} → PPL {cur_ppl:.2f}")

BEST_SUFFIX = tokenizer.decode(suffix_ids)
print(f"\n◎ 최종 universal suffix: {repr(BEST_SUFFIX)}  (PPL={cur_ppl:.2f})")
print(f"↳ 스텝 로그: {LOG_PATH.name}")

# ───── 결과 저장 ─────
for ex, base in zip(data, texts):
    ex["final_output"] = base + BEST_SUFFIX
RESULT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✔ 결과 저장 → {RESULT_JSON.resolve()}")
