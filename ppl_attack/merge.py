#open jsonl lowppl_each_L80_beam128.log_1_32.jsonl
import json
new_data = []
with open("lowppl_each_L80_beam128.log_1_32.jsonl", "r", encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['step'] == 80:
            new_data.append(data)
with open("lowppl_each_L80_beam128.log.jsonl", "r", encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['step'] == 80:
            new_data.append(data)

# Write the new data to a new json file not jsonl
#with open("lowppl_each_L80_beam128.log.json", "w", encoding='utf-8') as f:
#   json.dump(new_data, f, ensure_ascii=False, indent=4)
# Write the new data to a new jsonl file
me = 0
for item in new_data:
    me += item['ppl']
print(me / len(new_data))