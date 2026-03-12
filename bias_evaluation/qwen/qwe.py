from gradio_client import Client

# 간단한 API 엔드포인트 동작 확인 스크립트
client = Client("Qwen/Qwen3-Demo")

# 1) /new_chat 테스트
new_chat_res = client.predict(api_name="/new_chat")
print("new_chat_res:", new_chat_res, type(new_chat_res))

# 2) /add_message 테스트
settings = {
    "model": "qwen3-235b-a22b",
    "sys_prompt": "",        # 빈 프롬프트
    "thinking_budget": 1     # CoT 거의 off
}

# 단순 텍스트 입력 (I/O 포맷 없이)
raw_input = "Hello, this is a test."

# 위치 인자로 호출
add_res_pos = client.predict(raw_input, settings, api_name="/add_message")
print("\nadd_message (positional) raw result:", add_res_pos, type(add_res_pos))

# named 인자로 호출
add_res_named = client.predict(
    input_value=raw_input,
    settings_form_value=settings,
    api_name="/add_message"
)
print("\nadd_message (named) raw result:", add_res_named, type(add_res_named))

# 반환값 구조 확인
def inspect(res):
    if isinstance(res, tuple):
        print("Tuple len:", len(res))
        for i, e in enumerate(res):
            print(f"  [{i}] type:", type(e), "value:", e)
    else:
        print("Not a tuple, single output:", res)

print("\n-- Inspect positional call --")
inspect(add_res_pos)

print("\n-- Inspect named call --")
inspect(add_res_named)
