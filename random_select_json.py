import json
import random

# ==== 参数配置 ====
input_path = "/Users/huangxu/Downloads/inpo_iter2_skywork.json"   # 原始 19k JSON 文件路径
output_path = "../RL_agent_data_augment/inpo_iter2_skyworks_10k.json"  # 输出文件路径
sample_size = 10000          # 抽样数量
seed = 42                   # 固定随机种子，保证可复现

# ==== 读取 JSON 文件 ====
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ==== 随机抽样 ====
random.seed(seed)
sampled_data = random.sample(data, sample_size)

# ==== 保存结果 ====
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, indent=4, ensure_ascii=False)

print(f"✅ 已从 {len(data)} 条样本中随机抽取 {sample_size} 条，结果已保存至 {output_path}")
