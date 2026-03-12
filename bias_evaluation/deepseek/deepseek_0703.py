from openai import OpenAI
import csv
import uuid
import time
import json
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

with open('harmful_prompt.txt', 'r', encoding='utf-8') as f:
    harmful_prompt = f.read()
with open('animal_prompt.txt', 'r', encoding='utf-8') as f:
    animal_prompt = f.read()
    
with open('harmful_input.json', 'r', encoding='utf-8') as f:
    harmful_input = json.load(f)
with open('animal_input.json', 'r', encoding='utf-8') as f:
    animal_input = json.load(f)

# PROMPT = [[harmful_prompt, "PN"], [animal_prompt, "PA"]]
# INPUT = [[harmful_input, "IN"], [animal_input, "IA"]]

PROMPT = [[animal_prompt, "PA"]]
INPUT = [[animal_input, "IA"]]

# PROMPT = [[harmful_prompt, "PN"]]
# INPUT = [[harmful_input, "IN"]]

def ask_deepseek(user_input: str, is_free: bool, SYS_PROMPT) -> dict:
    # 1) 세션 초기화
    client = InferenceClient(
        provider="novita",
        api_key=os.getenv("hf_api"),
    )
    # 3) I/O 포맷 래핑
    wrapped = f"""<INPUT>
    <is_free_tier_user>{is_free}</is_free_tier_user>
    <user_input>{user_input}</user_input>
    </INPUT>"""
    print("im okay")
    completion = client.chat.completions.create(
        extra_body={},
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        messages=[
            {
            "role": "system",
            "content": SYS_PROMPT
            },
            {
            "role": "user",
            "content": wrapped
            }
        ],
        temperature=0.9,
        top_p=0.9
    )
    print(completion.choices[0].message.content)

    import re

    # <think> 내부 내용
    scratchpad_match = re.search(r"<think>(.*?)</think>", completion.choices[0].message.content, re.DOTALL)
    scratchpad = scratchpad_match.group(1).strip() if scratchpad_match else ""

    # </think> 이후 전체
    answer_match = re.search(r"</think>(.*)", completion.choices[0].message.content, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""


    return {
        "user_input": user_input,
        "is_free":    is_free,
        "scratchpad": scratchpad,
        "answer":     answer
    }

def run_batch(user_input, sys_prompt, output_path):
    fieldnames = ["uuid", "is_free", "user_input", "scratchpad", "answer", "sys_prompt"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for text in user_input: #toggle USER_INPUTS to test different sets ANIMAL / NORMAL
            for flag in (True, False):  # free-tier / paid-tier 쌍
                print(f"🔄 요청: {text} ({'Free' if flag else 'Paid'})")
                res = ask_deepseek(text, flag, sys_prompt)
                res["uuid"] = str(uuid.uuid4())
                res["sys_prompt"] = sys_prompt
                writer.writerow(res)
                f.flush() 
                time.sleep(0.5)  # 부하 방지
                print(f"✅ 완료: {res['uuid']} - {res['user_input']} ({'Free' if flag else 'Paid'})")

    print(f"✅ 완료: '{output_path}' 생성됨")

if __name__ == "__main__":
    filename = "deepseek_llama70B_results_0703.csv"
    for prompt, pna in PROMPT:
        for input_data, ina in INPUT:
            USER_INPUTS = input_data
            SYS_PROMPT = prompt
            print(f"Running with {pna} and {ina}...")
            run_batch(user_input = USER_INPUTS, sys_prompt = SYS_PROMPT, output_path=f"{pna}_{ina}_{filename}")
            print(f"Finished with {pna} and {ina}.")