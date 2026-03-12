import json
with open("train_data.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

import random
random.shuffle(data)

#split 6:2:2
train_data = data[:int(len(data) * 0.6)]
valid_data = data[int(len(data) * 0.6):int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):]

#save
with open("train.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
with open("valid.jsonl", "w", encoding="utf-8") as f:
    for item in valid_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
with open("test.jsonl", "w", encoding="utf-8") as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
