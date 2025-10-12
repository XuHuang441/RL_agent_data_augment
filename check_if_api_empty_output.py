import json
from pprint import pprint

file_path = "size10k/processed_samples_10k.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        for key,_ in data.items():
            print(key)
        break

print('-' * 100)

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # if i >= 3:  # 只看前三个
        #     break
        data = json.loads(line)
        if data.get('new_prompt') is None:
            pprint(data)
