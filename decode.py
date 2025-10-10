from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" # this is recommended for gemma-2 models; otherwise it is not needed
import argparse
import json
from tqdm import tqdm
from more_itertools import chunked

parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--data_dir', type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                    help='Directory containing the data')
parser.add_argument('--model', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--output_dir', type=str, default="datasets/gemma2_ultrafeedback",
                    help='output_dir')
parser.add_argument('--num_gpu', type=int, default=4)
parser.add_argument('--sanity_check', action='store_true', help="启用健全性检查（只用100个样本）")
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--cache_dir', type=str, default=None,
                    help='Cache directory for model and dataset')
args = parser.parse_args()

print(args)

data_dir = args.data_dir
llm = LLM(
    model=args.model,
    tensor_parallel_size=args.num_gpu,
    download_dir=args.cache_dir,
    gpu_memory_utilization=0.9,  # 允许 VLLM 使用 90% 的 GPU 显存
)
tokenizer = llm.get_tokenizer()

if os.path.exists(data_dir):
    # 如果输入是一个存在的本地文件路径
    print("检测到本地文件路径，正在加载本地文件...")
    # 使用 'json' 加载器，它同时支持 .json 和 .jsonl 文件
    train_dataset = load_dataset("json", data_files=data_dir, split="train")
else:
    # 如果不是本地文件路径，则假定它是一个Hugging Face Hub上的数据集名称
    print("未检测到本地文件，尝试从Hugging Face Hub加载...")
    train_dataset = load_dataset(data_dir, split="train")

# 如果是健全性检查，只选择少量样本
if args.sanity_check:
    print("执行健全性检查，仅使用100个样本。")
    train_dataset = train_dataset.select(range(min(len(train_dataset), 100)))

prompts = sorted(list(set(train_dataset['new_prompt'])))

conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

sampling_params = SamplingParams(temperature=args.temperature, 
                                 top_p=args.top_p, 
                                 max_tokens=args.max_tokens, 
                                 seed=args.seed,)
batched_prompts = list(chunked(conversations, args.batch_size))
output_data = []

for batch_prompts in tqdm(batched_prompts):
    try:
        outputs = llm.generate(batch_prompts, sampling_params)
        for i, output in enumerate(outputs):
            output_data.append({
                'prompt': batch_prompts[i],
                "format_prompt": output.prompt,
                'generated_text': output.outputs[0].text,
            })
    except Exception as e:
        print(f"Batch failed with error: {e}")

output_file = f'output_{args.seed}.json'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
