import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import DPOConfig, DPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    # I/O
    p.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--train_jsonl", type=str, required=True, help="本地 DPO 训练集 JSONL")
    p.add_argument("--eval_jsonl", type=str, required=True, help="本地 DPO 测试/验证集 JSONL")
    p.add_argument("--output_dir", type=str, default="./output/gemma2_dpo")
    p.add_argument("--cache_dir", type=str, default=None)

    # 核心超参
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=2048, help="prompt+answer 的最大长度")
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_target_length", type=int, default=1024)

    # 评估/保存/日志
    p.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"])
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=20)

    # 训练稳定性
    p.add_argument("--seed", type=int, default=191)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--grad_checkpointing", action="store_true")

    # 分布式/设备
    p.add_argument("--ddp_find_unused_parameters", action="store_true")
    p.add_argument("--tf32", action="store_true")

    # Weights & Biases
    p.add_argument("--wandb", action="store_true", help="是否上报到 wandb")
    p.add_argument("--wandb_project", type=str, default="dpo-local")
    p.add_argument("--wandb_run_name", type=str, default="gemma2-9b-it-dpo")

    # 可选量化/内存
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA 场景可开（需要 bitsandbytes）")
    p.add_argument("--load_in_8bit", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # 基础设置
    set_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # --- 数据集：本地 JSONL ---
    # 每行需包含：prompt / chosen / rejected（消息列表），以及 score_chosen / score_rejected（可选）
    data_files = {
        "train": args.train_jsonl,
        "validation": args.eval_jsonl,
    }
    ds = load_dataset("json", data_files=data_files)

    # --- 模型与分词器 ---
    # 注：大模型建议配合 --load_in_4bit 或 --load_in_8bit；或自行改为 LoRA Adapter
    model_kwargs = dict(cache_dir=args.cache_dir)
    if args.load_in_4bit:
        model_kwargs.update(dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16))
    elif args.load_in_8bit:
        model_kwargs.update(dict(load_in_8bit=True))

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)

    # Gemma/聊天模型：右填充更稳妥
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- DPO 配置 ---
    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.grad_checkpointing,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        # 报告后端
        report_to=("wandb" if args.wandb else "none"),
        run_name=(args.wandb_run_name if args.wandb else None),
    )

    # --- DPOTrainer ---
    # 说明：
    # - 你的 JSONL 是“消息列表”格式；TRL 会基于 tokenizer.chat_template 自动构造文本（对 Gemma-2 已内置模板）。
    # - max_length/max_prompt_length/max_target_length 控制截断，防止超长。
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_target_length=args.max_target_length,
    )

    # --- 可选：W&B 环境变量 ---
    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_RUN_GROUP"] = "dpo"
        # 也可额外设置 WANDB_API_KEY 在环境变量

    # --- 训练 ---
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
