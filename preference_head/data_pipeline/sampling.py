#!/usr/bin/env python3
import json, random

SRC = "arena_train.jsonl"
TEST_OUT = "pure_test.jsonl"
TRAIN_OUT = "for_train.jsonl"
SEED = 20250812  # 고정 시드

# 읽기
with open(SRC, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

if len(rows) < 3000:
    raise ValueError(f"데이터 {len(rows)}개. 최소 3000개 필요합니다.")

# 재현 가능한 셔플 후 분할(중복 없음)
rng = random.Random(SEED)
rng.shuffle(rows)

test_rows  = rows[:1000]
train_rows = rows[1000:3000]

# 저장
with open(TEST_OUT, "w", encoding="utf-8") as f:
    for r in test_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(TRAIN_OUT, "w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
