#!/usr/bin/env python
"""
universal_lowppl_responses_per_prompt.py
────────────────────────────────────────
• Alpaca‑Eval inst 셋 그대로 사용 (base response X)
• 100개 프롬프트 각각에 대해 길이 TARGET_LEN 토큰열 전역 Beam 탐색 → 해당 프롬프트의 PPL 최소
• 매 스텝 로그 및 중간 결과 JSON 저장 (atomic rename)
────────────────────────────────────────────────────────────────
pip install "transformers>=4.52.0" accelerate bitsandbytes torch tqdm
"""

import json, math, tempfile, datetime, torch
from pathlib import Path
from typing import List, Tuple
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
)

# ────────── 파라미터 ──────────
HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH   = "alpaca_eval.json"       # instruction·input 포함 원본 JSON
TARGET_LEN  = 80                       # “한 문장” 길이(토큰 수)
BEAM_WIDTH  = 128                      # 빔 폭
TOPK_EXPAND = 128                      # 각 노드에서 펼칠 상위 후보 수
BATCH_SIZE  = 16                       # PPL 계산용 배치(한 프롬프트 내 배치 없음·미사용)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

OUT_PREFIX  = f"lowppl_each_L{TARGET_LEN}_beam{BEAM_WIDTH}"
LOG_PATH    = Path(f"{OUT_PREFIX}.log.jsonl")
RESULT_PATH = Path(f"{OUT_PREFIX}.json")

# ────────── 유틸 ──────────
def atomic_save(path: Path, obj) -> None:
    tmp = Path(tempfile.mktemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix))
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)   # atomic

def log_step(rec: dict) -> None:
    rec["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ────────── 모델 로드 (4‑bit) ──────────
quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tok  = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=False)
mdl  = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, quantization_config=quant_cfg).to(DEVICE)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"

# ────────── Alpaca inst 프롬프트 준비 ──────────
raw = json.loads(Path(DATA_PATH).read_text())
prompts: List[str] = [
    f"### Instruction:\n{ex['instruction']}\n\n### Assistant:\n"
    for ex in raw
][:100]
prompt_tok_ids: List[List[int]] = [tok(p).input_ids for p in prompts]
print(f"총 프롬프트 수: {len(prompts)}")

# ────────── 평균 NLL(PPL) 계산 ──────────
@torch.no_grad()
def mean_nll_single(prompt_ids: List[int], seq_tok_ids: List[int]) -> float:
    """prompt_ids 뒤에 seq_tok_ids 를 붙였을 때 평균 NLL"""
    ids = prompt_ids + seq_tok_ids
    inp = torch.tensor(ids, device=DEVICE).unsqueeze(0)
    labels = inp.clone()
    labels[:, :len(prompt_ids)] = -100      # prompt 부분 마스킹
    loss = mdl(input_ids=inp, labels=labels).loss
    # 토큰 수(응답 부분) 계산
    tok_cnt = (labels != -100).sum().item()
    return loss.item() * 1.0  / tok_cnt

@torch.no_grad()
def expand_candidates(prompt_ids: List[int], prefix: List[int]) -> List[Tuple[int, float]]:
    """prompt+prefix 뒤 상위 TOPK_EXPAND 토큰 (id, logp)"""
    inp = torch.tensor(prompt_ids + prefix, device=DEVICE).unsqueeze(0)
    logits = mdl(inp).logits[0, -1]
    logps  = logits.log_softmax(dim=0)
    top = torch.topk(logps, TOPK_EXPAND)
    return [(tid.item(), logp.item()) for tid, logp in zip(top.indices, top.values)]

# ────────── 프롬프트별 Beam Search ──────────
LOG_PATH.unlink(missing_ok=True)
results = []

for idx, (p_txt, p_ids) in enumerate(zip(prompts, prompt_tok_ids), 1):
    if(idx < 33): continue  # 33번 프롬프트까지는 skip (예시용)
    if(idx > 100): break  # 100번 프롬프트까지만
    print(f"\n=== Prompt {idx}/{len(prompts)} ===")
    beam: List[Tuple[List[int], float]] = [([], 0.0)]  # (토큰열, joint logp)

    for step in range(1, TARGET_LEN + 1):
        cand_pool: List[Tuple[List[int], float]] = []
        for seq, lp_sum in beam:
            for tid, logp in expand_candidates(p_ids, seq):
                cand_pool.append((seq + [tid], lp_sum + logp))
        cand_pool.sort(key=lambda x: x[1], reverse=True)
        beam = cand_pool[:BEAM_WIDTH]

        best_seq, best_lp = beam[0]
        avg_nll = mean_nll_single(p_ids, best_seq)
        ppl = math.exp(avg_nll)

        log_step({
            "prompt_idx": idx,
            "step": step,
            "best_tokens": best_seq,
            "best_string": tok.decode(best_seq),
            "joint_logprob": best_lp,
            "avg_nll": avg_nll,
            "ppl": ppl,
        })

    best_seq = beam[0][0]
    best_text = tok.decode(best_seq)
    best_ppl  = math.exp(mean_nll_single(p_ids, best_seq))

    print(f"▶ 최종 PPL={best_ppl:.3f}  →  '{best_text.replace(chr(10),'↵')}'")

    results.append({
        "instruction": raw[idx-1]["instruction"],
        "input": raw[idx-1].get("input", ""),
        "final_output": best_text,
        "ppl": best_ppl,
    })

# ────────── 결과 저장 ──────────
atomic_save(RESULT_PATH, results)
print(f"\n↳ 단계별 로그: {LOG_PATH.name}")
print(f"↳ 결과 JSON : {RESULT_PATH.name}")
