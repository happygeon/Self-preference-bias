import json
with open("Greedy_Llama31-8B_chatstyle_outputs.json", "r", encoding="utf-8") as f:
    data = json.load(f)
with open("Greedy4CHK_Llama31-8B_chatstyle_outputs.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)

for item1, item2 in zip(data, data2):
    if item1["instruction"] != item2["instruction"]:
        print("Mismatch found!")
        print(f"Item1: {item1['instruction']}")
        print(f"Item2: {item2['instruction']}")
    if item1["output"] != item2["output"]:
        print("Mismatch found!")
        print(f"Item1: {item1['output']}")
        print(f"Item2: {item2['output']}")
    