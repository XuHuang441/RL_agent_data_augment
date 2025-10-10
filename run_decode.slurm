#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:4
#SBATCH --job-name=rl_agent_gen
#SBATCH --output=o_rlag.log
#SBATCH --mem=128G

export HF_HOME=/hai/scratch/fangwu97/xu/.cache/hf

#for SEED in 13 21 42 79 100
#  do
#     echo "Running decode with seed $SEED..."
#     stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m decode \
#     --data_dir "/hai/scratch/fangwu97/xu/RL_agent_data_augment/processed_samples.jsonl" \
#     --model "/hai/scratch/fangwu97/xu/cache/google/gemma-2-9b-it/" \
#     --seed "$SEED" \
#     --output_dir "/hai/scratch/fangwu97/xu/RL_agent_data_augment/model_gen" \
#     --batch_size 96 \
#     --cache_dir "/hai/scratch/fangwu97/xu/cache" \
#     --num_gpu 4 # Tensor Parallelism
#  done
#
#/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -m post_process \
#     --generation_file_dir "/hai/scratch/fangwu97/xu/RL_agent_data_augment/model_gen"

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -m reward_model_annotate_skywork \
     --generation_file "/hai/scratch/fangwu97/xu/RL_agent_data_augment/model_gen/all_outputs.json" \
     --output_dir "/hai/scratch/fangwu97/xu/RL_agent_data_augment/rm_skywork" \
     --reward_model "/hai/scratch/fangwu97/xu/cache/Skywork-Reward-V2-Llama-3.1-8B" \
     --cache_dir "/hai/scratch/fangwu97/xu/cache"

