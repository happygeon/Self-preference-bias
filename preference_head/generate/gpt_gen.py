
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
TEMPERATURE  = 0.0
MAX_TOKENS   = 512
OUT_PATH     = Path("chatbot_arena_chatgpt.jsonl")   # ★ 응답 저장 파일
SLEEP_SEC    = 1.0                     # API 호출 간 최소 간격
openai.api_key = os.environ.get("OPENAI_API_KEY")
# ────────────── 데이터 로더 ──────────────
def load_eval_set() -> List[Dict[str, Any]]:
    repo_id  = "tatsu-lab/alpaca_eval"
    filename = "alpaca_eval.json"
    path = hf_hub_download(repo_id=repo_id,
                           repo_type="dataset",
                           filename=filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ────────────── 보조 함수 ──────────────
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

def chat_once(instruction: str) -> str:
    """ChatGPT 호출, 오류 시 재시도"""
    while True:
        try:
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": instruction}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
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
def main() -> None:
    items = load_eval_set()
    done_idx = load_done_indices(OUT_PATH)

    for idx, item in enumerate(tqdm(items, desc="Generating"), 1):
        if idx in done_idx:
            continue  # 이미 처리됨

        instruction = item["instruction"]
        chat_reply = chat_once(instruction)

        rec = {
            "idx": idx,
            "dataset": item["dataset"],
            "instruction": instruction,
            "generator_input": item["output"],        # 원본 reference 답변
            "chatgpt_output": chat_reply,
            "model": MODEL_NAME,
        }
        append_record(OUT_PATH, rec)
        time.sleep(SLEEP_SEC)  # API 호출 간 최소 텀 확보

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] 중단됨. 다음 실행 시 이어서 처리됩니다.", file=sys.stderr)
