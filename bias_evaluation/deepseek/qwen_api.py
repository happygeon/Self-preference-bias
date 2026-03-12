from gradio_client import Client
import csv
import uuid
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()

# 원본에서 쓰던 파일 읽기 부분 그대로
with open('harmful_prompt.txt', 'r', encoding='utf-8') as f:
    harmful_prompt = f.read()
with open('animal_prompt.txt', 'r', encoding='utf-8') as f:
    animal_prompt = f.read()

with open('harmful_input.json', 'r', encoding='utf-8') as f:
    harmful_input = json.load(f)
with open('animal_input.json', 'r', encoding='utf-8') as f:
    animal_input = json.load(f)

PROMPT = [[harmful_prompt, "PN"], [animal_prompt, "PA"]]
INPUT = [[harmful_input, "IN"], [animal_input, "IA"]]

# PROMPT = [[animal_prompt, "PA"]]
# INPUT = [[animal_input, "IA"]]

MERGE = []
for i in range(len(PROMPT)):
    MERGE.append([PROMPT[i], INPUT[i]])

# Qwen Space 정보
SPACE_ID        = "Qwen/Qwen3-Demo"
THINKING_BUDGET = 37  # 필요에 따라 조정

def ask_deepseek(user_input: str, is_free: bool, SYS_PROMPT: str) -> dict:
    # 1) 세션 초기화
    client = Client(SPACE_ID)
    client.predict(api_name="/new_chat")

    # 2) 설정 (Qwen 에 맞춘 form 필드)
    settings = {
        "model":           "qwen3-14b",  # Space 내부에서 사용하는 모델명
        "sys_prompt":      SYS_PROMPT,
        "thinking_budget": THINKING_BUDGET
    }

    # 3) I/O 래핑(원본과 동일)
    wrapped = f"""<INPUT>
<is_free_tier_user>{is_free}</is_free_tier_user>
<user_input>{user_input}</user_input>
</INPUT>"""

    # 4) /add_message 호출
    reply, raw = client.predict(
        api_name="/add_message",
        input_value=wrapped,
        settings_form_value=settings
    )

    # 5) raw['value'] 파싱 (원본 코드와 동일)
    messages  = raw.get("value", [])
    scratchpad = ""
    answer     = ""

    for msg in messages:
        if msg.get("role") == "assistant":
            content_blocks = msg.get("content", [])
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if block.get("type") == "tool":
                        scratchpad += block.get("content", "") + "\n"
                    elif block.get("type") == "text":
                        answer = block.get("content", "")
            elif isinstance(content_blocks, str):
                answer = content_blocks

    return {
        "user_input": user_input,
        "is_free":    is_free,
        "scratchpad": scratchpad.strip(),
        "answer":     answer.strip()
    }

def run_batch(user_input, sys_prompt, output_path):
    fieldnames = ["uuid", "is_free", "user_input", "scratchpad", "answer", "sys_prompt"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for text in user_input:
            for flag in (True, False):
                print(f"🔄 요청: {text} ({'Free' if flag else 'Paid'})")
                res = ask_deepseek(text, flag, sys_prompt)
                res["uuid"]       = str(uuid.uuid4())
                res["sys_prompt"] = sys_prompt
                writer.writerow(res)
                f.flush()
                time.sleep(0.5)
                print(f"✅ 완료: {res['uuid']}")

    print(f"✅ 완료: '{output_path}' 생성됨")

if __name__ == "__main__":
    filename = "qwen14B_results_0703.csv"
    for P, I in MERGE:
        SYS_PROMPT, pna = P
        USER_INPUTS, ina = I
        print(f"Running with {pna} and {ina}...")
        run_batch(user_input = USER_INPUTS, sys_prompt = SYS_PROMPT, output_path=f"{pna}_{ina}_{filename}")
        print(f"Finished with {pna} and {ina}.")
    