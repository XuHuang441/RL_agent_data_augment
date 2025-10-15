#!/usr/bin/env python
import os
import json
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams
from tqdm import tqdm

from conversation import get_conv_template

# ===================== log =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== arguments =====================
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="INPO", metadata={"help": "Model identifier for result tracking."})
    model_path: Optional[str] = field(default="", metadata={"help": "Path to the model weights and tokenizer."})
    conv_temp: Optional[str] = field(default="myllama3", metadata={"help": "Conversation template name."})
    max_new_tokens: Optional[int] = field(default=4096, metadata={"help": "Maximum number of generated tokens."})
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed for reproducibility."})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "Sampling temperature."})
    use_beam_search: Optional[bool] = field(default=False, metadata={"help": "Enable beam search."})
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "End-of-sequence token IDs."})
    # ✨ NEW: Argument for number of GPUs
    tensor_parallel_size: Optional[int] = field(default=4, metadata={"help": "Number of GPUs to use."})
    cache_dir: Optional[str] = field(default="~/.cache/inpo", metadata={"help": "Directory to cache models and datasets."})


# ===================== main function =====================
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # random seed
    torch.manual_seed(script_args.seed)
    np.random.seed(script_args.seed)

    # output directory
    output_dir = "res"
    os.makedirs(output_dir, exist_ok=True)

    # initialize model
    logger.info(f"Loading model from: {script_args.model_path}")
    llm = LLM(
        model=script_args.model_path,
        tokenizer=script_args.model_path,
        dtype="float16",
        # ✨ CHANGE: Use all available GPUs for tensor parallelism
        tensor_parallel_size=script_args.tensor_parallel_size,
        max_model_len=script_args.max_new_tokens,
        load_format="auto",
        seed=script_args.seed,
        download_dir=script_args.cache_dir
    )

    # load evaluation dataset
    logger.info("Loading evaluation dataset...")
    eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]

    # set up sampling parameters
    conv_template = get_conv_template(script_args.conv_temp)
    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        top_p=1.0,
        max_tokens=script_args.max_new_tokens,
        n=1,
        stop_token_ids=conv_template.stop_token_ids or script_args.eos_ids
    )

    # ✨ CHANGE: Prepare all prompts in a batch
    logger.info(f"Preparing {len(eval_set)} prompts for batch generation...")
    prompts = []
    for sample in eval_set:
        conv = get_conv_template(script_args.conv_temp)
        conv.append_message(conv.roles[0], sample["instruction"])
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())

    # ✨ CHANGE: Generate answers for the entire batch at once
    logger.info("Starting batch generation...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # ✨ CHANGE: Process the results from the batch output
    answers = []
    for i, out in enumerate(outputs):
        sample = eval_set[i]
        generated_text = out.outputs[0].text
        answers.append({
            "instruction": sample["instruction"],
            "output": generated_text,
            "generator": script_args.model_name,
            "dataset": sample.get("dataset", "unknown")
        })

    answer_file = os.path.join(output_dir, f"{script_args.model_name}.json")
    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    logger.info(f"Generation complete. Results saved to: {answer_file}")