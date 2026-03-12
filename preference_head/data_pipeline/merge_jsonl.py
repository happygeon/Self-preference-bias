import json
path = '/home/happygeon02/new_dataset/dataset_json/'
def merge_jsonl(file):
    merged_data = []
    with open(path + file + '_alpaca.jsonl', 'r') as f1, open(path + file + '_chatbot_arena.jsonl', 'r') as f2:
        for line in f1:
            merged_data.append(json.loads(line))
        for line in f2:
            merged_data.append(json.loads(line))

    with open(path + file + '_merged.jsonl', 'w') as f:
        for item in merged_data:
            f.write(json.dumps(item) + '\n')

merge_jsonl('valid')
merge_jsonl('test')
merge_jsonl('train')
