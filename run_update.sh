export OPENAI_API_KEY="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"
python update_prompt_llm.py \
  --path inpo_iter2_skyworks_1000.json \
  --key all_rm_scores \
  --q 0.10 \
  --api_base https://openrouter.ai/api/v1 \
  --model gpt-5-mini \
  --out_jsonl left10_outputs.jsonl \
  --and_all_scores_le_global_mean \
   --test_run 3

