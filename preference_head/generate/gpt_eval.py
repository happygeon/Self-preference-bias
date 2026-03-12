
#!/usr/bin/env python
"""
batch_chatgpt_generate.py
─────────────────────────
• Alpaca-Eval → instruction별 ChatGPT 응답 생성
• 각 항목을 처리할 때마다 결과를 바로 <out_path>.jsonl 에 1줄(JSON)씩 append
  └ 이미 처리된 항목은 건너뜀(재시작 가능)
──────────────────────────────────────────────────────────────
환경:
  pip install openai transformers huggingface_hub tqdm
  export OPENAI_API_KEY="..."  # 또는 os.environ 에 키 세팅
"""

import json, os, sys, time
from pathlib import Path
from typing import List, Dict, Any, Set

import openai
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

# ────────────── 설정 ──────────────
MODEL_NAME   = "gpt-4o-mini"          # 필요 시 다른 ChatGPT 모델로 변경
TEMPERATURE  = 0
MAX_TOKENS   = 4
OUT_PATH     = "pairwise.jsonl"   # ★ 응답 저장 파일
SLEEP_SEC    = 0.15                     # API 호출 간 최소 간격
openai.api_key = os.environ.get("OPENAI_API_KEY")
# ────────────── 보조 함수 ──────────────
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
    
def load_done_indices(path: Path) -> Set[int]:
    """이미 저장된 레코드의 'idx' 집합을 반환"""
    done: Set[int] = set()
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        done.add(rec["idx"])
                    except Exception:
                        continue  # 손상된 줄 무시
    return done

def append_record(path: Path, record: Dict[str, Any]) -> None:
    """레코드를 한 줄(JSON)로 즉시 저장 (flush 포함)"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def chat_once(I: str, A: str, B: str) -> str:
    """ChatGPT 호출, 오류 시 재시도"""
    #random 1/2
    import random
    if random.randint(0, 1) == 0:
        instruction = build_pair_prompt(I, A, B)
        flag = False
    else:
        instruction = build_pair_prompt(I, B, A)
        flag = True
    
    while True:
        try:
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "SYSTEM_PROMPT = You are a helpful instruction-following assistant."},
                    {"role": "user", "content": instruction},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            
            res = resp.choices[0].message.content.strip()
            
            if flag:
                if res == "A":
                    final_answer = "b"
                    final_output = B
                elif res == "B":
                    final_answer = "a"
                    final_output = A
                else:
                    final_answer = res
                    final_output = "error"
            else:
                if res == "A":
                    final_answer = "a"
                    final_output = A
                elif res == "B":
                    final_answer = "b"
                    final_output = B
                else:
                    final_answer = res
                    final_output = "error"
            print("[INFO] ChatGPT response:", final_answer)
            return {
                "idx": 0,
                "flag": flag,
                "instruction": I,
                "label_a": A,
                "label_b": B,
                "response": res,
                "final_answer": final_answer,
                "final_output": final_output
            }
        
        
        except openai.error.RateLimitError:
            time.sleep(10)  # 과금제 제한 대비
        except openai.error.APIConnectionError:
            time.sleep(5)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[WARN] {e}", file=sys.stderr)
            time.sleep(2)

# ────────────── 메인 루프 ──────────────
def main(A: str, B: str) -> None:
    with open("chatbot_arena_llama_sample1.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open("chatbot_arena_llama_sample2.json", "r", encoding="utf-8") as f:
        data_b = json.load(f)
    OUT_PATH = Path(f"Asample1_vs_Bsample2"+"_pairwise.jsonl")

    done_idx = load_done_indices(OUT_PATH)

    i = 0
    for item, item_b in zip(data, data_b):
        idx = i
        label_a = item["output"]
        label_b = item_b["output"]
        instruction = item["instruction"]
        
        if idx in done_idx:
            continue  # 이미 처리됨

        chat_reply = chat_once(instruction, label_a, label_b)
        chat_reply["idx"] = idx
        append_record(OUT_PATH, chat_reply)
        time.sleep(SLEEP_SEC)  # API 호출 간 최소 텀 확보
        i += 1

if __name__ == "__main__":
    
    L1 = "15"
    L2 = "20"
    g = "results"
    try:
        #main(g, L1)
        #main(g, L2)
        #main(L1, L2)
        main(L1, L1)
    except KeyboardInterrupt:
        print("\n[INFO] 중단됨. 다음 실행 시 이어서 처리됩니다.", file=sys.stderr)
