#!/usr/bin/env python
"""
pairwise_eval_alpaca.py
───────────────────────
로컬 alpaca_eval.json × Llama‑3.1‑8B‑Instruct Judge
→ gold vs 공통 응답 승률 계산
──────────────────────────────────────────────────────
pip install "transformers>=4.52.0" accelerate bitsandbytes torch
"""

import json, random, re, torch, csv
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

# ────────── 설정 ──────────
ALPACA_PATH   = "llama31_alpaca_top100_with_prefix.json"   # 로컬 경로
JUDGE_ID      = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SEED    = 42                   # 재현용 시드
MAX_ITEMS     = 100                 # None = 전체, 숫자면 앞 N개만
OUT_CSV       = "pairwise_results.csv"

# 공통 응답(저‑PPL 문자열)을 정의하거나 읽어오세요
common_response =  "def is_prime(n):\n    \"\"\"Check if a number is prime.\"\"\"\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0"


# ────────── Judge 모델 로드 ──────────
tok = AutoTokenizer.from_pretrained(JUDGE_ID, use_fast=False)
mdl = AutoModelForCausalLM.from_pretrained(
    JUDGE_ID, load_in_4bit=True, device_map="auto"
).eval()

# ─── Judge 모델 파이프라인 (수정) ───
judge = pipeline(
    "text-generation",
    model=mdl,
    tokenizer=tok,
    max_new_tokens=10,
    do_sample=False,        # ✨ Greedy (sampling 꺼짐)
    temperature=0.0,        # 0.0 OK  – sampling 안 하면 무시됩니다
)
def build_pair_prompt(instruction: str, label_a: str, label_b: str) -> str:
    """
    ‘label_a’, ‘label_b’에 이미 섞어 둔 두 응답을 넣어,
    논문식 프롬프트 형태로 반환합니다.
    """
    return (
        "Select the output A or B that best matches the given instruction. "
        "Only reply 'A' or 'B'.\n\n"

        "# Example:\n"
        "## Instruction:\n"
        "Give a description of the following job: \"ophthalmologist\"\n\n"
        "## Output A:\n"
        "An ophthalmologist is a medical doctor who specializes in the diagnosis and treatment of eye diseases and conditions.\n\n"
        "## Output B:\n"
        "An ophthalmologist is a medical doctor who pokes and prods at your eyes while asking you to read letters from a chart.\n\n"
        "## Which is best, A or B?\n"
        "A\n\n"
        "Here the answer is A because it provides a comprehensive description.\n\n"

        "# Task:\n"
        "Do not explain, just say A or B.\n\n"

        "## Instruction:\n"
        f"{instruction}\n\n"
        "## Output A:\n"
        f"{label_a}\n\n"
        "## Output B:\n"
        f"{label_b}\n\n"
        "## Which is best, A or B?"
    )

# ────────── Pair‑wise 평가 함수 ──────────
def judge_pair(item_inst: str, gold: str, alt: str, rng: random.Random) -> str:
    pair = [gold, alt]
    rng.shuffle(pair)                           # A/B 섞기
    prompt = build_pair_prompt(item_inst, pair[0], pair[1])
    raw = judge(prompt)[0]["generated_text"][len(prompt):]
    m = re.search(r"\b(A|B|tie)\b", raw, re.I)
    if not m:
        return ("error", raw)
    pick = m.group(1).upper()
    if pick == "A":
        return ("gold", raw) if pair[0] == gold else ("alt", raw)
    if pick == "B":
        return ("gold", raw) if pair[1] == gold else ("alt", raw)
    return ("tie", raw)

# ────────── 데이터 로드 & 평가 ──────────
rng = random.Random(BATCH_SEED)
stats = {"gold": 0, "alt": 0, "tie": 0, "error": 0}

with Path(ALPACA_PATH).open(encoding="utf-8") as f:
    dataset = json.load(f)

if MAX_ITEMS:
    dataset = dataset[:MAX_ITEMS]

rows = []
for idx, ex in enumerate(dataset, 1):
    verdict, raw = judge_pair(ex["instruction"], ex["output"], ex["base_output"], rng)
    stats[verdict] += 1
    rows.append({"index": idx, "verdict": verdict, "raw": raw})
    if idx % 50 == 0:
        print(f"{idx}개 평가 완료…")

# ────────── 결과 요약 ──────────
total = sum(stats.values()) - stats["error"]
print("\n=== Pair‑wise 결과 (gold vs common) ===")
for k in ("gold", "alt", "tie", "error"):
    v = stats[k]
    print(f"{k:>5}: {v:4}")
print(f"총 평가: {total}개")
print(sum(stats.values()), "개 중 오류:", stats["error"])

# ────────── CSV 저장 (선택) ──────────
with Path(OUT_CSV).open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["index", "verdict", "raw"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\n세부 결과 저장 → {OUT_CSV}")
