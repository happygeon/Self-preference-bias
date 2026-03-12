import json

with open("filtered_pairs.jsonl", "r", encoding="utf-8") as f:
    data = []
    for line in f:
        data.append(json.loads(line))
print(f"Loaded {len(data)} records from filtered_pairs.jsonl")
length = len(data) // 2

ran = list(range(length))
#shuffle ran
import  random
random.shuffle(ran)

train = ran[200:]
valid = ran[100:200]
test = ran[:100]

#save jsonl
with open("train_chatbot_arena.jsonl", "w", encoding="utf-8") as f:
    for i in train:
        f.write(json.dumps(data[i * 2]) + "\n")
        f.write(json.dumps(data[i * 2 + 1]) + "\n")

with open("valid_chatbot_arena.jsonl", "w", encoding="utf-8") as f:
    for i in valid:
        f.write(json.dumps(data[i * 2]) + "\n")
        f.write(json.dumps(data[i * 2 + 1]) + "\n")

with open("test_chatbot_arena.jsonl", "w", encoding="utf-8") as f:
    for i in test:
        f.write(json.dumps(data[i * 2]) + "\n")
        f.write(json.dumps(data[i * 2 + 1]) + "\n")
