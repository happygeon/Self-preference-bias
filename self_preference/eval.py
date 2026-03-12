#!/usr/bin/env python
"""
unified_eval.py  
Pair‑wise A/B evaluator (point‑free) using Hugging Face **text‑generation** pipeline.  
- Batch size fixed at 27  
- No CLI – edit constants below.  
- Multi‑GPU (≥2) → `device_map="auto"` with optional per‑GPU memory limit.  
"""
from __future__ import annotations

import json, random, re, os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
# ================== 사용자 설정 ==================
FILE1 = "qwen25_watermelonhjg_outputs.json"      # 평가할 JSON 파일 1
FILE2 = "qwen25_xwen-team_outputs.json"      # 평가할 JSON 파일 2
MODEL1_LABEL = "watermelonhjg"           # file1 레이블
MODEL2_LABEL = "xwen"            # file2 레이블
JUDGE_MODEL = "xwen-team/Xwen-7B-Chat"  # 로컬 판정 모델 경로
DEVICE = "cuda:0"                 # GPU 디바이스
SEED = 42                           # 랜덤 시드
BATCH_SIZE = 12                     # 배치 사이즈
OUT_FILE = "watermelonhjg_vs_xwen_xwen_results.json"      # 결과 출력 파일

# ───────────────────────── 2. PROMPT TEMPLATE ─────────────────────
SYSTEM_PROMPT = "You are a helpful instruction‑following assistant."
USER_TMPL = (
    "Select the output A or B that best matches the given instruction. "
    "Only reply 'A' or 'B'.\n\n"
    "# Example:\n"
    "## Instruction:\nGive a description of the following job: \"ophthalmologist\"\n\n"
    "## Output A:\nAn ophthalmologist is a medical doctor who specializes in the diagnosis and treatment of eye diseases and conditions.\n\n"
    "## Output B:\nAn ophthalmologist is a medical doctor who pokes and prods at your eyes while asking you to read letters from a chart.\n\n"
    "## Which is best, A or B?\nA\n\n"
    "Here the answer is A because it provides a comprehensive description.\n\n"
    "# Task:\nDo not explain, just say A or B.\n\n"
    "## Instruction:\n{instruction}\n\n"
    "## Output A:\n{output_a}\n\n"
    "## Output B:\n{output_b}\n\n"
    "## Which is best, A or B?"
)

# Preferred pattern for extracting winner
PREF_RE = re.compile(r"\b([ab]|1|2|tie|draw)\b", re.I)
TIE_SET = {"tie", "draw"}

# ───────────────────────── 3. UTILITIES ───────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def random_swap(row: pd.Series, rng: np.random.Generator) -> Tuple[pd.Series, Dict[str, str], bool]:
    """Randomly swap outputs 50 % of the time to prevent side bias."""
    if rng.random() < 0.5:
        return row, {"1": "a", "2": "b"}, False
    row2 = row.copy()
    row2["output_a"], row2["output_b"] = row["output_b"], row["output_a"]
    return row2, {"1": "b", "2": "a"}, True

# ───────────────────────── 4. PIPELINE LOADER ─────────────────────

def load_text_pipeline(model_name: str):
    """Create 8‑bit `text‑generation` pipeline with multi‑GPU support."""
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # device map
    if torch.cuda.device_count() > 1:
        device_map = "auto"
        max_memory = {i: "22GiB" for i in range(torch.cuda.device_count())}
    else:
        device_map = {"": 0 if torch.cuda.is_available() else "cpu"}
        max_memory = None

    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        torch_dtype="auto",
        device_map=device_map,
        **({"max_memory": max_memory} if max_memory else {})
    )
    mdl.eval()

    return pipeline(
        task="text-generation",
        model=mdl,
        tokenizer=tok,
        batch_size=BATCH_SIZE,
        return_full_text=False,
    )

# ───────────────────────── 5. PARSE WINNER ────────────────────────

def parse_pref(txt: str) -> str:
    m = PREF_RE.search(txt.lower())
    if not m:
        return "error"
    g = m.group(1)
    if g in {"1", "a"}:
        return "1"
    if g in {"2", "b"}:
        return "2"
    if g in TIE_SET:
        return "tie"
    return "error"

# ───────────────────────── 6. EVALUATION CORE ─────────────────────

def evaluate(file1: str, file2: str) -> List[Dict[str, Any]]:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    # Load JSON → DataFrame
    d1, d2 = map(lambda f: json.load(open(f, encoding="utf-8")), (file1, file2))
    if len(d1) != len(d2):
        raise ValueError("JSON 길이가 다릅니다.")
    df1 = pd.DataFrame(d1).reset_index().rename(columns={"index": "orig_idx"})
    df2 = pd.DataFrame(d2)
    df_pair = df1.join(df2["output"].rename("output_b"))
    df_pair.rename(columns={"output": "output_a"}, inplace=True)

    # Shuffle & random swap
    df_pair = df_pair.sample(frac=1, random_state=SEED).reset_index(drop=True)
    rows, mappings, swaps = [], [], []
    for _, row in df_pair.iterrows():
        r2, m, sw = random_swap(row, rng)
        rows.append(r2); mappings.append(m); swaps.append(sw)

    # Build prompts
    prompts = [
        SYSTEM_PROMPT + "\n\n" + USER_TMPL.format(
            instruction=r["instruction"],
            output_a=r["output_a"],
            output_b=r["output_b"],
        )
        for r in rows
    ]

    # Run generation
    gen_pipe = load_text_pipeline(JUDGE_MODEL)
    raw_outs = gen_pipe(prompts, max_new_tokens=256, do_sample=False)

    # Extract answers
    outs = [o[0]["generated_text"].strip() if isinstance(o, list) else o["generated_text"].strip() for o in raw_outs]
    print("✅  판정 모델 출력:", outs)
    # Collate results
    results: List[Dict[str, Any]] = []
    for r, mp, o, sw in zip(rows, mappings, outs, swaps):
        pref = parse_pref(o)
        if pref == "1":
            win_letter = "a" if mp["1"] == "a" else "b"
        elif pref == "2":
            win_letter = "a" if mp["2"] == "a" else "b"
        elif pref == "tie":
            win_letter = "tie"
        else:
            win_letter = "error"
        win_model = (
            MODEL1_LABEL if win_letter == "a" else MODEL2_LABEL if win_letter == "b" else win_letter
        )
        results.append({
            "instruction": r["instruction"],
            "output_a": r["output_a"],
            "model_a": MODEL1_LABEL,
            "output_b": r["output_b"],
            "model_b": MODEL2_LABEL,
            "winner": win_letter,
            "winner_model": win_model,
            "rand_swapped": sw,
            "debugging output": o,
        })
    return sorted(results, key=lambda d: d.get("orig_idx", 0))

# ───────────────────────── 7. MAIN ───────────────────────────────
if __name__ == "__main__":
    res = evaluate(FILE1, FILE2)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"✅  평가 완료 → {OUT_FILE} ({len(res)} records)")
