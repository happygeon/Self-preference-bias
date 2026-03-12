import json

li = []
def preprocess(file_name: str):
    with open(file_name, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    li = []
    for item in data:
        tmp = {}
        tmp["q"] = item["instruction"]
        tmp["win"] = item["final_output"]
        
        a = item["label_a"]
        b = item["label_b"]
        if item["final_answer"] == "a":
            tmp["lose"] = b
        else:
            tmp["lose"] = a
        li.append(tmp)
    return li


li += preprocess("1Llama31-8B_output_2Llama31-8B_output_pairwise.jsonl")
li += preprocess("chatgpt_output_1Llama31-8B_output_pairwise.jsonl")
li += preprocess("chatgpt_output_2Llama31-8B_output_pairwise.jsonl")


with open("train_data.jsonl", "w", encoding="utf-8") as f:
    for item in li:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
