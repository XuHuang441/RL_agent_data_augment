import json

def filter_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = []
    for sample in data:
        scores = sample.get("all_rm_scores", [])
        if not scores:  # 无分数，跳过
            continue

        # 条件 1: 所有分数相等
        if all(s == scores[0] for s in scores):
            continue

        # 条件 2: 所有分数都小于 0
        if all(s < 0 for s in scores):
            continue

        filtered.append(sample)

    print(f"原始样本数: {len(data)}, 保留样本数: {len(filtered)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=4)

# 示例调用
filter_json("1k/all_outputs_rm.json", "1k/filtered.json")
