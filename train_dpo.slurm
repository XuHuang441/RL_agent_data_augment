#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:4
#SBATCH --job-name=rl_agent_gen
#SBATCH --output=o_rlag.log
#SBATCH --mem=128G

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m train_dpo_local \
    --train_jsonl dpo_data.train.jsonl \
    --eval_jsonl  dpo_data.test.jsonl \
    --output_dir ./output/gemma2_dpo \
    --model_name /hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/ \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --beta 10 \
    --max_length 2048 \
    --max_prompt_length 1800 \
    --evaluation_strategy steps \
    --eval_steps 1 \
    --save_strategy steps \
    --save_steps 2 \
    --save_total_limit 2 \
    --load_best_model_at_end: True \
    --metric_for_best_model: "eval_loss" \
    --logging_steps 1 \
    --seed 42 \
    --bf16 \
    --grad_checkpointing

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m train_dpo2 \
    --dataset_name po_ready_data \
    --model_name_or_path /hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/ \
    --learning_rate 5.0e-7 \
    --beta 0.01 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns