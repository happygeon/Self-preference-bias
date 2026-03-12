import json

with open("base_vs_inst_inst_results.json", "r", encoding='utf-8') as file:
    data = file.read()

json_data = json.loads(data)

dict = {}
for item in json_data:
    tmp = item['winner_model']
    if tmp not in dict:
        dict[tmp] = 1
    else:
        dict[tmp] += 1
        
for key, value in dict.items():
    print(f"{key}: {value}번")