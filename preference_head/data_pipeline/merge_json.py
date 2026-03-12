import json

with open('1Llama31-8B_chatstyle_outputs.json', 'r') as file:
    data_1 = json.load(file)
with open('2Llama31-8B_chatstyle_outputs.json', 'r') as file:
    data_2 = json.load(file)
with open('alpaca_eval_chatgpt.jsonl', 'r') as file:
    data_g = [json.loads(line) for line in file]

print(f"1Llama31-8B: {len(data_1)}")
print(f"2Llama31-8B: {len(data_2)}")
print(f"ChatGPT: {len(data_g)}")

li = []
for i1, i2, ig in zip(data_1, data_2, data_g):
    idx = ig['idx']
    print(f"Processing idx: {idx}")
    inst = ig['instruction']
    if i1['instruction'] != inst:
        print(f"Mismatch at idx {idx}: 1Llama31-8B: {i1['instruction']}, 2Llama31-8B: {i2['instruction']}, ChatGPT: {inst}")
    if i2['instruction'] != inst:
        print(f"Mismatch at idx {idx}: 2Llama31-8B: {i2['instruction']}, 1Llama31-8B: {i1['instruction']}, ChatGPT: {inst}")
    ground = ig['generator_input']
    og = ig['chatgpt_output']
    o1 = i1['output']
    o2 = i2['output']
    
    tmp = {
        'idx': idx,
        'instruction': inst,
        'ground': ground,
        'chatgpt_output': og,
        '1Llama31-8B_output': o1,
        '2Llama31-8B_output': o2,
    }
    li.append(tmp)
    
    if o1 == o2 or o1 == og or o2 == og:
        print(f"Match found at idx {idx}: {o1} == {o2} or {o1} == {og} or {o2} == {og}")
breakpoint()  # 디버깅용
with open('merged_outputs.json', 'w') as file:
    json.dump(li, file, indent=2, ensure_ascii=False)
    