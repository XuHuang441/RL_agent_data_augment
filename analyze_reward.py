import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

file_path = "merged_final_data.json"
# file_path = "size1k/inpo_iter2_skyworks_1000.json"
# file_path = "/Users/huangxu/Downloads/inpo_iter2_skywork.json"

# 假设文件名为 data.json
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 如果文件里是多个对象，可以遍历
all_scores = []
for item in data:
    if "all_rm_scores" in item:
        all_scores.extend(item["all_rm_scores"])

# 转 numpy 数组便于统计
scores = np.array(all_scores)

# 基本统计信息
print("总数:", len(scores))
print("最小值:", np.min(scores))
print("最大值:", np.max(scores))
print("均值:", np.mean(scores))
print("中位数:", np.median(scores))

# 分布统计（计数）
counter = Counter(scores)
print("频数分布:")
for k, v in sorted(counter.items()):
    print(f"  {k}: {v}")

# 绘制直方图
plt.hist(scores, bins=20, edgecolor="black")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Distribution of all_rm_scores")
plt.show()
