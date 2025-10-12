import json

# 读取第一个 JSON 文件
with open("size1k/high_quality_samples.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

# 读取第二个 JSON 文件
with open("rm_skywork/1k/filtered.json", "r", encoding="utf-8") as f2:
    data2 = json.load(f2)

# 合并两个列表
merged = data1 + data2

# 保存到新文件
with open("merged_final_data.json", "w", encoding="utf-8") as out:
    json.dump(merged, out, ensure_ascii=False, indent=2)

print(f"合并完成，共 {len(merged)} 条数据。")
