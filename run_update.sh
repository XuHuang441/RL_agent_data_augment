python update_prompt_llm.py \
  --path inpo_iter2_skyworks_1000.json \
  --key all_rm_scores \
  --q 0.10 \
  --api_base https://openrouter.ai/api/v1 \
  --model gpt-5-mini \
  --out_jsonl left10_outputs.jsonl \
  --and_all_scores_le_global_mean \
   --test_run 3

