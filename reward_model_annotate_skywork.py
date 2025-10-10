import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import argparse
import tqdm
import numpy as np
import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file", type=str, default="datasets/gemma2_ultrafeedback/all_outputs.json",
                    help="Path to the output generation file")
# CHANGED: Updated the default model name
parser.add_argument("--reward_model", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                    help="Path to reward model")
parser.add_argument("--output_dir", type=str, default="datasets/gemma2_ultrafeedback/", help="Path to output directory")
parser.add_argument('--cache_dir', type=str, default=None,
                    help='Cache directory for model and dataset')
args = parser.parse_args()

print(args)

generation_file = args.generation_file
with open(generation_file, 'r') as f:
    output_data = json.load(f)

inputs = [data["prompt"] for data in output_data]
candidates_texts = [data["all_generated_responses"] for data in output_data]

# --- MODIFICATION 1: Model Loading ---
# Added attn_implementation and num_labels as recommended in the example
model = AutoModelForSequenceClassification.from_pretrained(
    args.reward_model,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    cache_dir=args.cache_dir,
    attn_implementation="flash_attention_2",  # Recommended for performance
    num_labels=1,  # Standard for reward models
)
tokenizer = AutoTokenizer.from_pretrained(args.reward_model, cache_dir=args.cache_dir)

for data in tqdm.tqdm(output_data):
    prompt = data["prompt"]
    candidates = data["all_generated_responses"]
    scores = []
    for candidate in candidates:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": candidate}]

        # --- MODIFICATION 2: Tokenization Process ---
        # The new model requires a two-step tokenization process
        # Step 1: Apply template to get a string
        formatted_input = tokenizer.apply_chat_template(messages, tokenize=False)

        # Optional but good practice: remove potential duplicate bos token
        if tokenizer.bos_token is not None and formatted_input.startswith(tokenizer.bos_token):
            formatted_input = formatted_input[len(tokenizer.bos_token):]

        # Step 2: Tokenize the formatted string
        input_ids = tokenizer(formatted_input, return_tensors="pt").to("cuda")

        with torch.no_grad():
            # The model expects keyword arguments, so we unpack the dictionary
            output = model(**input_ids)
            # --- MODIFICATION 3: Score Extraction ---
            # Score is in .logits[0][0] instead of .score
            score = output.logits[0][0].float().item()
            scores.append(score)

    data["all_rm_scores"] = scores

file_name = os.path.basename(args.generation_file).split('.json')[0] + "_rm.json"
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, file_name), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Annotated outputs saved to {os.path.join(args.output_dir, file_name)}")

# Binarize data: win = highest scoring reponse; lose = lowest scoring response
for data in output_data:
    chosen_idx = np.argmax(data["all_rm_scores"])
    rejected_idx = np.argmin(data["all_rm_scores"])
    chosen = []
    chosen.append({
        "role": "user",
        "content": data["prompt"]
    })
    chosen.append({
        "role": "assistant",
        "content": data["all_generated_responses"][chosen_idx]
    })
    rejected = []
    rejected.append({
        "role": "user",
        "content": data["prompt"]
    })
    rejected.append({
        "role": "assistant",
        "content": data["all_generated_responses"][rejected_idx]
    })
    data.update({
        "chosen": chosen,
        "rejected": rejected,
    })

# --- BUG FIX: Saving to the correct file ---
# The original script was overwriting the _rm.json file. This now saves to _bin.json as intended.
output_file = os.path.basename(args.generation_file).split('.json')[0] + "_bin.json"
with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)
print(f"Binarized outputs saved to {os.path.join(args.output_dir, output_file)}")

# Convert the data to Hugging Face datasets format
dataset_output_path = os.path.join(args.output_dir,
                                   os.path.basename(args.generation_file).split('.json')[0] + "_bin_dataset")
dataset = datasets.Dataset.from_list(output_data)
dataset.save_to_disk(dataset_output_path)
print(f"Binarized dataset saved to {dataset_output_path}")