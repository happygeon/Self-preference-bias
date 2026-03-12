#!/usr/bin/env python
"""
batch_chatgpt_generate.py
─────────────────────────
• 로컬 JSONL(각 줄: {"instruction": ...}) → instruction별 ChatGPT 응답 생성
• 각 항목을 처리할 때마다 결과를 바로 <out_path>.jsonl 에 1줄(JSON)씩 append
  └ 이미 처리된 항목은 건너뜀(재시작 가능)
──────────────────────────────────────────────────────────────
환경:
  pip install openai tqdm
  export OPENAI_API_KEY="..."  # 환경변수로 API 키 지정
"""

import json, os, sys, time, argparse
from pathlib import Path
from typing import List, Dict, Any, Set

import openai
from tqdm.auto import tqdm

# ────────────── 기본 설정 (CLI로 덮어씀) ──────────────
MODEL_NAME  = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS  = 512
SLEEP_SEC   = 0.15

# ────────────── 유틸 ──────────────
def load_local_jsonl(path: Path) -> List[Dict[str, Any]]:
    """각 줄이 {"instruction": "..."} 인 JSONL 파일을 리스트로 로드"""
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 파싱 오류 {path}:{ln}: {e}") from e
            inst = obj.get("instruction", "")
            if not isinstance(inst, str) or not inst.strip():
                continue
            items.append({"instruction": inst})
    if not items:
        raise ValueError(f"유효한 instruction 레코드가 없습니다: {path}")
    return items

def load_done_indices(path: Path) -> Set[int]:
    """이미 저장된 레코드의 'idx' 집합을 반환"""
    done: Set[int] = set()
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if "idx" in rec:
                            done.add(int(rec["idx"]))
                    except Exception:
                        continue  # 손상된 줄 무시
    return done

def append_record(path: Path, record: Dict[str, Any]) -> None:
    """레코드를 한 줄(JSON)로 즉시 저장 (flush + fsync)"""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def chat_once(instruction: str, model: str, temperature: float, max_tokens: int) -> str:
    """ChatGPT 호출, 오류 시 재시도"""
    while True:
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": instruction}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            time.sleep(10)
        except openai.error.APIConnectionError:
            time.sleep(5)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[WARN] {e}", file=sys.stderr)
            time.sleep(2)

# ────────────── 메인 ──────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path, help="로컬 JSONL 경로 (각 줄: {\"instruction\": ...})")
    ap.add_argument("--out", type=Path, default=Path("chatbot_arena_chatgpt.jsonl"), help="출력 JSONL 경로")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--sleep", type=float, default=SLEEP_SEC, help="API 호출 간 최소 간격(초)")
    args = ap.parse_args()

    # API 키
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

    items = load_local_jsonl(args.jsonl)
    done_idx = load_done_indices(args.out)

    for idx, item in enumerate(tqdm(items, desc="Generating"), 1):
        if idx in done_idx:
            continue  # 이미 처리됨

        instruction = item["instruction"]
        reply = chat_once(
            instruction=instruction,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        rec = {
            "idx": idx,
            "instruction": instruction,
            "chatgpt_output": reply,
            "model": args.model,
        }
        append_record(args.out, rec)
        time.sleep(args.sleep)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] 중단됨. 다음 실행 시 이어서 처리됩니다.", file=sys.stderr)
