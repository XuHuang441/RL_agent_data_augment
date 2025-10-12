python process_pref_data.py \
--in_json merged_final_data.json --out_jsonl po_ready_data.jsonl \
--do_split --test_ratio 0.10 \
--hf_format --out_dir po_ready_data
