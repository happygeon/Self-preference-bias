#!/usr/bin/env python
"""
batch_chatstyle_generate.py
───────────────────────────
* Alpaca‑Eval QA 세트 → Llama‑3.1‑8B‑Instruct Chat‑style 응답 생성
* 8‑bit 양자화 + device_map="auto"
* datasets 로딩, 배치‑생성, JSON 저장
──────────────────────────────────────────────────────────────
pip install transformers accelerate bitsandbytes huggingface_hub tqdm torch datasets
"""

import json
from math import ceil
from typing import List, Dict

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
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
    """여러 GPU에 8‑bit 모델 로드 후 (model, tokenizer) 반환"""
    qconfig = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("CUDA 가용 GPU가 없습니다.")

    total_mem_gib = torch.cuda.get_device_properties(0).total_memory // 2**30
    max_memory = {i: f"{total_mem_gib-1}GiB" for i in range(n_gpu)}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=qconfig,
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()
    return model, tokenizer

# ────────────── 데이터셋 로더 ──────────────
def load_eval_ds() -> Dataset:
    json_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        repo_type="dataset",
        filename="alpaca_eval.json",
    )
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    processed: List[Dict] = [
        {"text": ex.get("instruction") or ex.get("question") or ""}
        for ex in raw
    ]
    return Dataset.from_list(processed)

# ────────────── 채팅 템플릿 적용 + 배치 토크나이즈 ──────────────
def build_batch_inputs(prompts: List[str], tokenizer) -> torch.Tensor:
    """프롬프트 리스트 → chat template 적용한 input_ids(batch)"""
    messages_batch = [
        [
            {"role": "user",   "content": p},
        ]
        for p in prompts
    ]

    # ① 각 프롬프트를 Chat 템플릿으로 문자열化
    prompt_texts = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False )
        for m in messages_batch
    ]
    # ② 토크나이저가 직접(left) 패딩 + attention_mask 생성
    batch = tokenizer(
        prompt_texts,
        padding=True,                # longest 시퀀스 길이에 맞춤
        return_tensors="pt",
    )
    return batch

# ────────────── 추론 ──────────────
def generate_responses(model, tokenizer, eval_ds: Dataset) -> List[dict]:
    prompts: List[str] = eval_ds["text"]
    outputs: List[dict] = []

    total_batches = ceil(len(prompts) / BATCH_SIZE)
    for idx in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[idx : idx + BATCH_SIZE]
        inputs = build_batch_inputs(batch_prompts, tokenizer).to(model.device)

        batch = build_batch_inputs(batch_prompts, tokenizer).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                **batch,                    # input_ids, attention_mask 모두 전달
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )


        for prompt, out_ids, inp_ids in zip(batch_prompts, gen_ids, batch["input_ids"]):
            gen_only = out_ids[len(inp_ids):]                # ← PAD 제외한 길이만큼 스킵
            text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

            if text == "":
                breakpoint()  # 디버깅용
            outputs.append({"instruction": prompt, "output": text})
        print(f"[{idx+len(batch_prompts):>6}/{len(prompts)}] 완료")

    return outputs

# ────────────── 메인 ──────────────
def main():
    eval_ds = load_eval_ds()
    print(f"Loaded {len(eval_ds)} prompts.")

    models = {
        "1Llama31-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
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
