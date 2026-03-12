import json

with open("base_vs_inst_base_results.json", "r", encoding='utf-8') as file:
    data = file.read()
with open("base_vs_inst_inst_results.json", "r", encoding='utf-8') as file2:
    data2 = file2.read()
json_data = json.loads(data)
json_data2 = json.loads(data2)


spb = 0
rspb = 0
idx = 0
for item, item2 in zip(json_data, json_data2):
    idx += 1
    if item['winner'] == "error" or item2['winner'] == "error":
        continue
    if item['winner'] == item2['winner']:
        continue
    
    if item['winner'] == "a" and item2['winner'] == "b":
        spb += 1
    if item['winner'] == "b" and item2['winner'] == "a":
        rspb += 1

print(f"Base vs Inst: {spb} vs {rspb}")
print(f"Total comparisons: {idx}")