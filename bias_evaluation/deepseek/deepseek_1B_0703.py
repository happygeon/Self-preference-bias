import csv
import uuid
import time
import json
import os
from llama_cpp import Llama
from dotenv import load_dotenv
load_dotenv()


llm = Llama(
	model_path="DeepSeek-R1-0528-UD-IQ1_S-00001-of-00004.gguf",
    n_gpu_layers=45,          # 전체 레이어 GPU
    tensor_split=[0.5, 0.5],  # GPU-0·1 VRAM 절반씩
    main_gpu=0,
    n_ctx=1024,
)


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

def ask_deepseek(user_input: str, is_free: bool, SYS_PROMPT) -> dict:
    # 1) 세션 초기화
    # 3) I/O 포맷 래핑
    wrapped = f"""<INPUT>
    <is_free_tier_user>{is_free}</is_free_tier_user>
    <user_input>{user_input}</user_input>
    </INPUT>"""
    print("im okay")
    completion = llm.create_chat_completion(
        messages=[
            {
            "role": "system",
            "content": SYS_PROMPT
            },
            {
            "role": "user",
            "content": wrapped
            }
        ]
    )
    print(completion.choices[0].message.content)
    print("=====================  Output format  ====================")
    import re

    scratchpad = re.search(r"(<SCRATCHPAD_REASONING>.*?</SCRATCHPAD_REASONING>)", completion.choices[0].message.content, re.DOTALL)
    scratchpad = scratchpad.group(1).strip() if scratchpad else ""
    print(scratchpad)
    print("=====================  End Of Reasoning format  ====================")
    answer_match = re.search(r"</SCRATCHPAD_REASONING>(.*)", completion.choices[0].message.content, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else completion.choices[0].message.content
    print(answer)
    print("=====================  End Of Output format  ====================")
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
    filename = "deepseek_IQ1_S_0703.csv"
    for P, I in MERGE:
        SYS_PROMPT, pna = P
        USER_INPUTS, ina = I
        print(f"Running with {pna} and {ina}...")
        run_batch(user_input = USER_INPUTS, sys_prompt = SYS_PROMPT, output_path=f"{pna}_{ina}_{filename}")
        print(f"Finished with {pna} and {ina}.")
    