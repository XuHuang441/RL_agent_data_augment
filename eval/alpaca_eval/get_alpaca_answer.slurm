#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=eval
#SBATCH --output=output_eval.log
#SBATCH --mem=128G

model_name="gemma-2-9b-it-DPO-10k-original"
model_path="/hai/scratch/fangwu97/xu/RL_agent_data_augment/gemma-2-9b-it-DPO-10k-original"
# CUDA_VISIBLE_DEVICES=3 python get_alpaca_answer.py --model_name $model_name --model_path $model_path --conv_temp "gemma"

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/RL_agent_data_augment/eval/alpaca_eval/get_alpaca_answer_fast.py \
    --model_name $model_name \
    --model_path $model_path \
    --conv_temp "gemma" \
    --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/outputs" \
    --tensor_parallel_size 2



