#!/usr/bin/env python
"""
pairwise_eval_alpaca_gpt4o.py
─────────────────────────────
로컬 alpaca_eval.json × GPT‑4o‑mini Judge (OpenAI API)
→ gold vs 공통 응답 승률 계산
────────────────────────────────────────────────────────
pip install openai==1.* python-dotenv
"""

import json, random, re, csv, time, os, sys
from pathlib import Path
from typing import Tuple, Dict, Any

import openai
from dotenv import load_dotenv

# ────────── 환경 설정 ──────────
load_dotenv()                                     # .env에 OPENAI_API_KEY 저장해 두었다면 읽어옵니다.
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    sys.exit("❌  OPENAI_API_KEY 환경 변수가 없습니다.")

ALPACA_PATH = "llama31_alpaca_top100_with_prefix.json"   # 원본 데이터셋
GEN_PATH    = "lowppl_each_L80_beam128.log.json"         # 추가 생성 데이터
JUDGE_MODEL = "gpt-4o-mini"                              # OpenAI Judge 모델
BATCH_SEED  = 42                                         # 시드 고정
MAX_ITEMS   = 100                                        # None = 전부
OUT_CSV     = "pairwise_results.csv"

# ────────── 프롬프트 빌더 ──────────
def build_pair_prompt(instruction: str, label_a: str, label_b: str) -> str:
    """A/B 두 응답을 섞어 평가용 프롬프트 문자열 생성"""
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

# ────────── GPT‑4o 호출 래퍼 ──────────
def chat_complete(prompt: str, model: str = JUDGE_MODEL,
                  max_tokens: int = 4,
                  retries: int = 3,
                  timeout: int = 30) -> str:
    """OpenAI ChatCompletion → 문자열 반환 (재시도 포함)"""
    for attempt in range(1, retries + 1):
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                top_p=1.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries:
                raise
            sleep_s = 2 ** attempt
            print(f"⚠️  {e} – {sleep_s}s 후 재시도({attempt}/{retries})…")
            time.sleep(sleep_s)

# ────────── Pair‑wise 평가 ──────────
def judge_pair(inst: str, gold: str, alt: str,
               rng: random.Random) -> Tuple[str, str]:
    """
    gold vs alt → 'gold' / 'alt' / 'tie' / 'error', 원본 응답 텍스트 반환
    """
    pair = [gold, alt]
    rng.shuffle(pair)                                # A/B 섞기

    prompt = build_pair_prompt(inst, pair[0], pair[1])
    raw = chat_complete(prompt)

    m = re.search(r"\b(A|B|tie)\b", raw, re.I)
    if not m:
        return "error", raw

    pick = m.group(1).upper()
    if pick == "A":
        return ("gold" if pair[0] == gold else "alt"), raw
    if pick == "B":
        return ("gold" if pair[1] == gold else "alt"), raw
    return "tie", raw

# ────────── 데이터 로드 ──────────
rng = random.Random(BATCH_SEED)
with Path(ALPACA_PATH).open(encoding="utf-8") as f:
    dataset = json.load(f)
with Path(GEN_PATH).open(encoding="utf-8") as f:
    gen_data = json.load(f)
if MAX_ITEMS:
    dataset = dataset[:MAX_ITEMS]

# ────────── 평가 루프 ──────────
stats: Dict[str, int] = {"gold": 0, "alt": 0, "tie": 0, "error": 0}
rows: list[Dict[str, Any]] = []

for idx, ex in enumerate(dataset, 1):
    tmp = gen_data[idx - 1]

    verdict, raw = judge_pair(
        ex["instruction"],
        ex["base_output"],        # gold (Alpaca GT)
        tmp['best_string'],   # alt (모델 또는 공통 응답)
        rng,
    )
    stats[verdict] += 1
    rows.append({"index": idx, "verdict": verdict, "raw": raw})

    if idx % 25 == 0:
        print(f"{idx}개 완료…")

# ────────── 결과 요약 ──────────
total_valid = stats["gold"] + stats["alt"] + stats["tie"]
print("\n=== Pair‑wise 결과 (gold vs alt) ===")
for k in ("gold", "alt", "tie", "error"):
    print(f"{k:>5}: {stats[k]:4}")
print(f"유효 평가: {total_valid} / {sum(stats.values())}")

# ────────── CSV 저장 ──────────
with Path(OUT_CSV).open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["index", "verdict", "raw"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n세부 결과 저장 → {OUT_CSV}")
