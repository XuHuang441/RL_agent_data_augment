import json

file_path = "processed_samples.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # if i >= 3:  # 只看前三个
        #     break
        data = json.loads(line)
        print(data['model_raw_response'])
