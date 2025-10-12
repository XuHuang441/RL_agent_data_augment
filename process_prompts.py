import json
import os
import sys

import numpy as np
from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict, Any
import concurrent.futures
from functools import partial

SANITY_CHECK = False
test_num = 10
# --- 1. 配置 ---

# 从环境变量中获取 OpenAI API 密钥
# 建议使用环境变量以避免在代码中硬编码密钥
# 如果您没有设置环境变量，也可以直接在此处赋值: api_key="sk-..."
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
if not api_key:
    raise ValueError("未找到 OpenAI API 密钥。请设置 OPENAI_API_KEY 环境变量。")

# 初始化 OpenAI 客户端
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

# 文件路径配置
INPUT_JSON_FILE = 'inpo_iter2_skyworks_10k.json'  # 输入的 JSON 文件名
OUTPUT_JSONL_FILE = 'size10k/processed_samples_10k.jsonl'  # 输出的 JSONL 文件名
HIGH_QUALITY_JSON_FILE = 'size10k/high_quality_samples_10k.json'

# 模型和参数配置
OPENAI_MODEL = "gpt-5-mini"  # 推荐使用支持 JSON 模式的较新模型
QUANTILE = 0.10  # 定义“左侧q分位数”的q值，例如0.10代表最低的10%

# --- 2. 任务模板 ---

# 这个模板将用于指示大模型执行指定的任务
# {original_prompt} 会被实际的低分 prompt 替换
TASK_PROMPT_TEMPLATE = """
You are an expert in AI training data generation and prompt engineering. Your task is to help me create a high-quality data pair for a challenging scenario where my own models are failing.

**1. The Original Difficult Prompt:**
Please analyze the following prompt, which my AI assistants consistently fail to answer correctly.

Original failed prompt:

{original_prompt}

**2. Your Task:**
Based on the prompt above, please perform the following two steps:

* **Step 1: Analyze:** Briefly explain why this prompt is challenging. What specific skills does it test (e.g., complex reasoning, creativity, following negative constraints, deep domain knowledge)?
* **Step 2: Generate a New Prompt:** Create one (1) *single-question* prompt that is conceptually similar to the original. It should test the same core skills you identified, but be phrased differently to increase data diversity.  
  - Do **not** include multiple questions or multiple examples.  
  - Do **not** simply reword the original prompt.  
  - Ensure the new prompt is self-contained and clearly instructs the model on what to do.

**3. Output Format:**
Please structure your response using the following JSON format:

{{
    "analysis": "<Your analysis of the original prompt's difficulty>",
    "new_prompt": "<a single new, well-formed prompt>"
}}
"""



def load_and_filter_samples(filepath: str, q: float) -> List[Dict[str, Any]]:
    """
    从 .json 文件加载样本，计算平均分，并筛选出低分样本。

    筛选标准:
    1. 样本的平均分位于所有样本平均分的左侧 q 分位数。
    2. 样本的所有单项分数都不高于整体样本的平均分。
    """
    print(f"正在从 {filepath} 读取数据...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：输入文件 {filepath} 未找到。")
        return []
    except json.JSONDecodeError:
        print(f"错误：文件 {filepath} 不是有效的 JSON 格式。")
        return []

    if not data:
        print("警告：输入文件为空。")
        return []

    # 计算每个样本的平均分和所有分数的列表
    all_scores = []
    for sample in data:
        scores = sample.get('all_rm_scores', [])
        if scores:
            sample['avg_score'] = np.mean(scores)
            all_scores.extend(scores)
        else:
            sample['avg_score'] = None  # 标记没有分数的样本

    # 过滤掉没有分数的样本
    valid_samples = [s for s in data if s['avg_score'] is not None]
    if not valid_samples:
        print("错误：数据中没有找到任何有效的评分。")
        return []

    # 计算整体统计数据
    overall_avg_score = np.mean(all_scores)
    avg_scores = [s['avg_score'] for s in valid_samples]
    score_threshold = np.quantile(avg_scores, q)

    print(f"总共加载了 {len(valid_samples)} 个有效样本。")
    print(f"所有分数的整体平均分: {overall_avg_score:.4f}")
    print(f"样本平均分的 {q * 100:.0f}% 分位数阈值: {score_threshold:.4f}")

    # 筛选低分样本
    low_quality_samples = []
    high_quality_samples = []
    for sample in valid_samples:
        is_low_quantile = sample['avg_score'] <= score_threshold
        all_scores_below_avg = all(score <= overall_avg_score for score in sample['all_rm_scores'])

        # 同时满足两个条件才算低分样本
        if is_low_quantile and all_scores_below_avg:
            low_quality_samples.append(sample)
        else:
            # 否则就是高分样本
            high_quality_samples.append(sample)

    print(f"筛选出 {len(low_quality_samples)} 个低分样本。")
    print(f"剩余 {len(high_quality_samples)} 个高分样本。")

    # 返回两个列表
    return low_quality_samples, high_quality_samples


def query_openai_for_enhancement(prompt: str) -> Dict[str, Any]:
    """调用 OpenAI API 来分析和增强给定的 prompt。"""
    try:
        # 构建发送给模型的 prompt
        system_prompt = TASK_PROMPT_TEMPLATE.format(original_prompt=prompt)

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": system_prompt}
            ],
            # 使用 JSON 模式，确保模型返回有效的 JSON
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        return {"raw_response": response.choices[0].message.content, "error": None}
    except Exception as e:
        # 捕获 API 调用过程中可能出现的任何错误
        return {"raw_response": None, "error": str(e)}


def parse_model_output(raw_response: str) -> Dict[str, Any]:
    """尝试从模型的原始文本输出中解析 JSON 结构。"""
    if not raw_response:
        return {"status": "failure_empty_response", "data": {}}

    try:
        # 尝试解析 JSON
        parsed_data = json.loads(raw_response)

        # 提取核心内容
        analysis = parsed_data.get("analysis", "N/A")
        new_prompt = parsed_data.get("new_prompt", "N/A")
        reference_answer = parsed_data.get("reference_answer", "N/A")

        return {
            "status": "success",
            "data": {
                "analysis": analysis,
                "new_prompt": new_prompt,
                "reference_answer": reference_answer
            }
        }
    except json.JSONDecodeError:
        # 如果解析失败，记录状态
        return {"status": "failure_json_decode", "data": {}}


def process_single_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个样本的完整流程：调用API、解析、构建结果。
    这是一个辅助函数，便于并行化。
    """
    original_prompt = sample.get('prompt')
    if not original_prompt:
        return None

    # 调用 OpenAI API
    api_result = query_openai_for_enhancement(original_prompt)

    # 准备要写入文件的结果字典
    output_record = {
        "original_prompt": original_prompt,
        "original_scores": sample.get('all_rm_scores'),
        "model_raw_response": api_result["raw_response"],
        "api_error": api_result["error"],
    }

    # 解析模型返回的 JSON
    if api_result["raw_response"]:
        parsed_result = parse_model_output(api_result["raw_response"])
        output_record["parse_status"] = parsed_result["status"]
        output_record.update(parsed_result["data"])
    else:
        output_record["parse_status"] = "failure_api_call"

    return output_record


def main():
    """主执行函数 (使用 ThreadPoolExecutor 进行并行处理)"""
    # 步骤 1: 加载并筛选样本
    low_quality_samples, high_quality_samples = load_and_filter_samples(INPUT_JSON_FILE, QUANTILE)

    if high_quality_samples:
        try:
            # 在写入前，我们可以选择性地移除为计算添加的 'avg_score' 字段
            # 这样可以保持输出文件与原始文件格式完全一致
            for sample in high_quality_samples:
                sample.pop('avg_score', None)

            with open(HIGH_QUALITY_JSON_FILE, 'w', encoding='utf-8') as f:
                # 使用 indent=4 格式化输出 JSON，使其易于阅读
                json.dump(high_quality_samples, f, indent=4, ensure_ascii=False)
            print(f"\n高分样本已成功保存至: {HIGH_QUALITY_JSON_FILE}")
        except IOError as e:
            print(f"错误：无法写入文件 {HIGH_QUALITY_JSON_FILE}。原因: {e}")

    if not low_quality_samples:
        print("没有需要处理的样本，程序退出。")
        return

    if SANITY_CHECK:
        print(f"sanity check enabled: only {test_num} samples will be processed")
        low_quality_samples = low_quality_samples[:test_num]
    else:
        print("sanity check disabled: use all samples")

    MAX_WORKERS = 10

    print(f"\n开始并行处理 {len(low_quality_samples)} 个样本 (使用 {MAX_WORKERS} 个工作线程)...")

    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as outfile:
        # 使用 ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 使用 executor.map 来并行执行 process_single_sample 函数
            # tqdm 用于显示进度条
            results_iterator = executor.map(process_single_sample, low_quality_samples)

            for result in tqdm(results_iterator, total=len(low_quality_samples), desc="Processing Samples"):
                if result:
                    # 将结果写入 .jsonl 文件
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\n处理完成！所有结果已成功保存至 {OUTPUT_JSONL_FILE}。")


if __name__ == '__main__':
    # 执行主程序
    main()