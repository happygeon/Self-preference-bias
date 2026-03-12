#!/usr/bin/env python
"""
batch_chatstyle_generate.py
───────────────────────────
* 로컬 JSONL(각 줄: {"instruction": ...}) → Llama-3.1-8B-Instruct Chat-style 응답 생성
* 8-bit 양자화 + device_map="auto"
* 배치-생성, JSON 저장
──────────────────────────────────────────────────────────────
pip install transformers accelerate bitsandbytes huggingface_hub tqdm torch datasets
"""

import json
from math import ceil
from typing import List, Dict

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download  # 남겨둠(원하면 삭제 가능)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ────────────── 하이퍼파라미터 ──────────────
BATCH_SIZE     = 32
MAX_NEW_TOKENS = 512
SYSTEM_PROMPT  = "You are a helpful assistant."   # 필요 시 변경

# ────────────── 모델 로더 ──────────────
def load_model(model_id: str):
    """여러 GPU에 8-bit 모델 로드 후 (model, tokenizer) 반환"""
    qconfig = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("CUDA 가용 GPU가 없습니다.")

    total_mem_gib = torch.cuda.get_device_properties(0).total_memory // 2**30
    max_memory = {i: f"{max(total_mem_gib-1, 1)}GiB" for i in range(n_gpu)}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=qconfig,
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()
    return model, tokenizer

# ────────────── 로컬 JSONL 로더 ──────────────
def load_local_jsonl(path: str) -> Dataset:
    """각 줄이 {"instruction": "..."} 인 JSONL을 읽어 {"text": "..."}로 변환"""
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 파싱 오류 {path}:{ln}: {e}") from e
            inst = obj.get("instruction", "")
            if isinstance(inst, str) and inst.strip():
                records.append({"text": inst})
    if not records:
        raise ValueError("유효한 instruction 레코드가 없습니다.")
    return Dataset.from_list(records)

# ────────────── 채팅 템플릿 적용 + 배치 토크나이즈 ──────────────
def build_batch_inputs(prompts: List[str], tokenizer) -> Dict[str, torch.Tensor]:
    """프롬프트 리스트 → chat template 적용한 input_ids/attention_mask"""
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    prompt_texts = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in messages_batch
    ]
    batch = tokenizer(prompt_texts, padding=True, return_tensors="pt")
    return batch

# ────────────── 추론 ──────────────
def generate_responses(model, tokenizer, eval_ds: Dataset) -> List[dict]:
    prompts: List[str] = eval_ds["text"]
    outputs: List[dict] = []

    total = len(prompts)
    for idx in range(0, total, BATCH_SIZE):
        batch_prompts = prompts[idx: idx + BATCH_SIZE]
        batch = build_batch_inputs(batch_prompts, tokenizer)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,           # 샘플링 활성화
                temperature=1.0,          # 0.7 같은 값 설정 (낮을수록 결정적)
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for prompt, out_ids, inp_ids in zip(batch_prompts, gen_ids, batch["input_ids"]):
            gen_only = out_ids[len(inp_ids):]
            text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
            outputs.append({"instruction": prompt, "output": text})

        print(f"[{min(idx+len(batch_prompts), total):>6}/{total}] 완료", flush=True)

    return outputs

# ────────────── 메인 ──────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="로컬 JSONL 경로 (각 줄: {\"instruction\": ...})")
    ap.add_argument("--model_id", default="/datasets2/Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    eval_ds = load_local_jsonl(args.jsonl)
    print(f"Loaded {len(eval_ds)} prompts from {args.jsonl}.")

    models = {
        "1Llama31-8B": args.model_id,
    }

    for name, repo in models.items():
        print(f"\n=== {name} ===")
        model, tokenizer = load_model(repo)
        responses = generate_responses(model, tokenizer, eval_ds)

        out_path = f"{name.replace('/', '_')}_chatstyle_outputs.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
        print(f"Saved → {out_path}")

if __name__ == "__main__":
    main()
