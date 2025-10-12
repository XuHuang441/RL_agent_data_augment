[analyze_reward.py](analyze_reward.py)

分析：总数
最小值
最大值
均值
中位数
频数分布并作图

random_select_json.py

首先从19k的一个skywork评测的数据集中随机选取自定义数量的sample作为测试（暂定10k）
[inpo_iter2_skyworks_10k.json](inpo_iter2_skyworks_10k.json)

[process_prompts.py](process_prompts.py)

输入：[inpo_iter2_skyworks_10k.json](inpo_iter2_skyworks_10k.json)
，“左侧q分位数”的q值（0.1）

首先根据rm score定位到平均分数最低的prompt（low_quality_samples），用teacher生成替换prompt

输出：
1. 剔除掉low_quality_samples，剩下的prompts：[high_quality_samples.json](size1k%2Fhigh_quality_samples.json)
2. teacher生成的prompts：[processed_samples.jsonl](size1k%2Fprocessed_samples.jsonl)

[check_if_api_empty_output.py](check_if_api_empty_output.py)

检查processed_samples.jsonl中有没有api返回空的输出

[run_decode.slurm](run_decode.slurm)

使用模型，用[processed_samples.jsonl](size1k%2Fprocessed_samples.jsonl)里的prompt生成5个新回复，并使用rm标注每个回复的分数