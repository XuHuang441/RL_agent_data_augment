"""
Select left-tail (q fraction) samples by mean(all_rm_scores),
extract their prompts, send to an OpenAI-compatible LLM with a provided template,
and save results to JSONL.

Usage example:
python left_tail_llm_generate.py \
  --path data.jsonl \
  --key all_rm_scores \
  --prompt_key prompt \
  --q 0.10 \
  --api_base http://127.0.0.1:8000/v1 \
  --model qwen2.5-32b-instruct \
  --out_jsonl left10_outputs.jsonl \
  --max_concurrency 8
"""

import os
import json
import argparse
from pathlib import Path
from statistics import mean
import numpy as np
import asyncio
import time
import math
from ast import literal_eval
from typing import Any, Dict, List, Tuple, Optional

try:
    import httpx  # 用于异步HTTP调用
except ImportError:
    raise SystemExit("Please `pip install httpx` first.")

USER_PROMPT_TEMPLATE = """
You are an expert in AI training data generation and prompt engineering. Your task is to help me create a high-quality data pair for a challenging scenario where my own models are failing.

**1. The Original Difficult Prompt:**
Please analyze the following prompt, which my AI assistants consistently fail to answer correctly.

Original failed prompt:

{original_prompt}

**2. Your Task:**
Based on the prompt above, please perform the following three steps:

* **Step 1: Analyze:** Briefly explain why this prompt is challenging. What specific skills does it test (e.g., complex reasoning, creativity, following negative constraints, deep domain knowledge)?
* **Step 2: Generate a New Prompt:** Create one (1) new prompt that is conceptually similar to the original. It should test the same core skills you identified, but be phrased differently to increase data diversity. Do not simply reword the original.
* **Step 3: Provide an Ideal Answer:** Write a gold-standard, ideal answer for the **new prompt** you just created. This answer should be:
    * Accurate, clear, and comprehensive.
    * Well-structured and easy to read.
    * Fully aligned with all instructions and constraints in the new prompt.
    * Written from the perspective of a helpful and knowledgeable AI assistant.

**3. Output Format:**
Please structure your response using the following JSON format:

{{
    "analysis": "<Your analysis of the original prompt's difficulty>",
    "new_prompt": "<rewritten or similar improved prompt>",
    "reference_answer": "<high-quality example answer to the new prompt.>"
}}
""".strip()

def load_records(path: Path) -> List[Dict[str, Any]]:
    """支持 JSON(数组) 或 JSONL(每行一个对象)"""
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return data


def get_by_path(obj: Dict[str, Any], dotted: str, default=None):
    """支持点号路径（如 'data.meta.prompt'）读取"""
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def select_left_tail(
    records: List[Dict[str, Any]],
    score_key: str,
    q: float,
    and_all_scores_le_global_mean: bool = False,
) -> Tuple[List[int], np.ndarray, float, float]:
    """按 per-object mean 的左侧 q 分位选样本。
    若启用 and_all_scores_le_global_mean，则仅保留所有分数 ≤ 全局均值的样本。
    返回 (选中索引, 均值数组, 分位阈值, 全局均值)
    """
    all_scores = []
    per_means, all_le_flags = [], []

    # 计算全局均值
    for obj in records:
        scores = obj.get(score_key, [])
        if isinstance(scores, list):
            all_scores.extend(scores)
    if not all_scores:
        raise ValueError("No scores found to compute global mean.")
    global_mean = sum(all_scores) / len(all_scores)

    # 每个样本的均值、是否所有分数≤全局均值
    for obj in records:
        scores = obj.get(score_key, [])
        if isinstance(scores, list) and len(scores) > 0:
            per_means.append(mean(scores))
            all_le_flags.append(all(s <= global_mean for s in scores))
        else:
            per_means.append(float("nan"))
            all_le_flags.append(False)

    means = np.array(per_means, dtype=float)
    flags = np.array(all_le_flags, dtype=bool)
    valid_mask = ~np.isnan(means)
    valid_means = means[valid_mask]

    thr = np.quantile(valid_means, q)
    selected_idx = [
        i
        for i, (m, ok, all_le) in enumerate(zip(means, valid_mask, flags))
        if ok and m < thr and (not and_all_scores_le_global_mean or all_le)
    ]
    return selected_idx, means, thr, global_mean


async def call_chat_completions(
    client: httpx.AsyncClient,
    api_base: str,
    model: str,
    api_key: str,
    messages: List[Dict[str, str]],
    timeout: float = 60.0,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    """调用 OpenAI 兼容 Chat Completions API，返回 assistant content 字符串"""
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # "response_format": {"type": "json_object"},
    }
    if model == "gpt-5-mini":
        payload["reasoning_effort"] = "low"

    resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # 兼容标准字段
    content = data["choices"][0]["message"]["content"]
    return content


def safe_json_parse(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """尝试把 LLM 输出解析为 JSON 对象；失败则返回 (None, 错误信息)"""

    attempts: List[str] = []
    MAX_NESTED_DEPTH = 3

    def _coerce_to_dict(value: Any, depth: int) -> Optional[Dict[str, Any]]:
        """Normalize parsed values (possibly strings/lists) into a dict when possible."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                if depth >= MAX_NESTED_DEPTH:
                    attempts.append("nested JSON string exceeded max depth")
                    return None
                return _attempt(stripped, depth + 1)
        if isinstance(value, list) and len(value) == 1:
            if depth >= MAX_NESTED_DEPTH:
                attempts.append("nested list wrapper exceeded max depth")
                return None
            return _coerce_to_dict(value[0], depth + 1)
        attempts.append(f"parsed value of type {type(value).__name__} is not a dict")
        return None

    def _attempt(candidate: str, depth: int = 0) -> Optional[Dict[str, Any]]:
        """Try tolerant parsers on the candidate string."""
        try:
            loaded = json.loads(candidate)
        except Exception as json_err:
            attempts.append(f"json.loads: {json_err}")
        else:
            maybe_dict = _coerce_to_dict(loaded, depth)
            if maybe_dict is not None:
                return maybe_dict
        try:
            parsed = literal_eval(candidate)
        except Exception as literal_err:
            attempts.append(f"literal_eval: {literal_err}")
            return None
        maybe_dict = _coerce_to_dict(parsed, depth)
        if maybe_dict is not None:
            return maybe_dict
        return None

    s = s.strip()
    if not s:
        return None, "empty response"

    # 有些模型会包一层 ```json ... ```
    if s.startswith("```"):
        fence = "```"
        s = s.strip(fence).strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()

    # 首先尝试整体解析
    parsed = _attempt(s)
    if parsed is not None:
        return parsed, None

    # 再尝试截取首尾的大括号之间的部分
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        parsed = _attempt(candidate)
        if parsed is not None:
            return parsed, None

    # 部分模型会在 JSON 后补充说明，如 "...}" + "某些额外内容"
    # 尝试逐步收缩尾部，直到能成功解析或耗尽
    if start != -1:
        for end_idx in range(len(s) - 1, start, -1):
            if s[end_idx] == "}":
                candidate = s[start : end_idx + 1]
                parsed = _attempt(candidate)
                if parsed is not None:
                    return parsed, None

    error = "; ".join(dict.fromkeys(attempts)) if attempts else "failed to parse JSON"
    return None, error


DESIRED_OUTPUT_KEYS = ("analysis", "new_prompt", "reference_answer")


def extract_desired_fields(parsed: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Return a dict with the required keys mapped to strings or None."""
    result: Dict[str, Optional[str]] = {key: None for key in DESIRED_OUTPUT_KEYS}
    if not isinstance(parsed, dict):
        return result

    for key in DESIRED_OUTPUT_KEYS:
        value = parsed.get(key)
        if isinstance(value, str):
            result[key] = value
        elif value is not None:
            # Convert non-string JSON values into a compact JSON string for logging.
            try:
                result[key] = json.dumps(value, ensure_ascii=False)
            except TypeError:
                result[key] = str(value)
    return result


async def producer_consumer(
    tasks: List[Tuple[int, Dict[str, Any]]],
    api_base: str,
    model: str,
    api_key: str,
    out_jsonl: Path,
    max_concurrency: int = 8,
    retry: int = 3,
    backoff: float = 2.0,
):
    sem = asyncio.Semaphore(max_concurrency)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # 文件是同步上下文管理器，单独用 with
        with out_jsonl.open("w", encoding="utf-8") as fout:

            async def _one(idx: int, payload: Dict[str, Any]):
                async with sem:
                    last_err = None
                    for attempt in range(1, retry + 1):
                        try:
                            response_text = await call_chat_completions(
                                client=client,
                                api_base=api_base,
                                model=model,
                                api_key=api_key,
                                messages=[
                                    {"role": "user", "content": payload["user_prompt_text"]},
                                ],
                                timeout=180.0,
                            )
                            llm_json, parse_error = safe_json_parse(response_text)
                            llm_struct = llm_json
                            desired_fields = extract_desired_fields(llm_json)
                            fout.write(json.dumps({
                                "index": idx,
                                "original_prompt": payload.get("original_prompt"),
                                "llm_raw": response_text,
                                "llm_struct": llm_struct,
                                "llm_struct_parse_error": parse_error,
                                "analysis": desired_fields["analysis"],
                                "new_prompt": desired_fields["new_prompt"],
                                "reference_answer": desired_fields["reference_answer"],
                                "sample_meta": payload.get("sample_meta", {}),
                                "ts": time.time(),
                            }, ensure_ascii=False) + "\n")
                            # 及时落盘（可选）
                            fout.flush()
                            return
                        except Exception as e:
                            last_err = e
                            if attempt < retry:
                                await asyncio.sleep(backoff ** attempt)
                    # 最终失败也写入一条错误记录，便于排查
                    fout.write(json.dumps({
                        "index": idx,
                        "original_prompt": payload.get("original_prompt"),
                        "error": str(last_err),
                        "sample_meta": payload.get("sample_meta", {}),
                        "ts": time.time(),
                    }, ensure_ascii=False) + "\n")
                    fout.flush()

            await asyncio.gather(*[_one(i, p) for i, p in tasks])


def main():
    ap = argparse.ArgumentParser(description="Select left q by mean(all_rm_scores), send prompts to LLM, save JSONL.")
    ap.add_argument("--path", required=True, help="Path to .json or .jsonl")
    ap.add_argument("--key", default="all_rm_scores", help="Score key (default: all_rm_scores)")
    ap.add_argument("--prompt_key", default="prompt", help="Prompt key path (supports dotted path). Default: prompt")
    ap.add_argument("--q", type=float, default=0.10, help="Left quantile in (0,1). Default: 0.10")
    ap.add_argument("--api_base", required=True, help="OpenAI-compatible API base, e.g. https://api.openai.com/v1 or http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="Model name to call")
    ap.add_argument("--out_jsonl", default="left_q_outputs.jsonl", help="Output JSONL file for LLM results")
    ap.add_argument("--max_concurrency", type=int, default=8, help="Max concurrent requests")
    ap.add_argument("--dry_run", action="store_true", help="Only print selected prompts; do not call API")
    ap.add_argument("--test_run", type=int, nargs="?", const=3,
                    help="发送前N条样本到API(默认3)，用于测试完整流程。")
    ap.add_argument("--and_all_scores_le_global_mean", action="store_true",
                    help="Require all single scores <= global mean")
    args = ap.parse_args()

    if not (0.0 < args.q < 1.0):
        raise ValueError("--q must be in (0,1)")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not args.dry_run and not api_key:
        raise SystemExit("Please set environment variable OPENAI_API_KEY for API auth, or use --dry_run to skip calls.")

    records = load_records(Path(args.path))
    selected_idx, means, thr, global_mean = select_left_tail(
        records,
        score_key=args.key,
        q=args.q,
        and_all_scores_le_global_mean=args.and_all_scores_le_global_mean,
    )
    total = np.count_nonzero(~np.isnan(means))
    print(f"[Info] Total valid samples: {total}")
    print(f"[Info] Global mean: {global_mean:.6f}")
    print(f"[Info] q={args.q:.3f} threshold mean<{thr:.6f}")
    print(f"[Info] Selected count: {len(selected_idx)} ({len(selected_idx) / total * 100:.2f}%)")

    # 构造任务：提取 prompt，拼模板
    tasks: List[Tuple[int, Dict[str, Any]]] = []
    missing_prompt = 0
    for idx in selected_idx:
        obj = records[idx]
        original_prompt = get_by_path(obj, args.prompt_key, default=None)
        if not isinstance(original_prompt, str):
            missing_prompt += 1
            original_prompt = ""  # 占位，依然发送，以便定位问题

        user_prompt_text = USER_PROMPT_TEMPLATE.format(original_prompt=original_prompt)

        sample_meta = {
            "mean": means[idx] if not math.isnan(means[idx]) else None,
            "index": idx,
        }
        tasks.append((idx, {
            "original_prompt": original_prompt,
            "user_prompt_text": user_prompt_text,
            "sample_meta": sample_meta,
        }))

    print(f"[Info] prompts found: {len(tasks)}  (missing prompt fields: {missing_prompt})")

    if args.dry_run:
        # 只打印几个示例
        print("\n=== DRY RUN PREVIEW (first 3) ===")
        for i, payload in tasks[:3]:
            print(f"- index={i}, mean={payload['sample_meta']['mean']}")
            print(payload["user_prompt_text"][:400] + ("..." if len(payload["user_prompt_text"]) > 400 else ""))
        return

    if args.test_run:
        n = min(args.test_run, len(tasks))
        print(f"\n=== Test Run: sending first {n} samples ===")
        asyncio.run(producer_consumer(
            tasks[:n],
            api_base=args.api_base,
            model=args.model,
            api_key=api_key,
            out_jsonl=Path(args.out_jsonl),
            max_concurrency=1,  # 顺序执行，方便调试
        ))
        print(f"[Info] Test run done, saved to {args.out_jsonl}")
        return

    # 并发调用 API 并保存
    out_jsonl = Path(args.out_jsonl)
    asyncio.run(producer_consumer(
        tasks=tasks,
        api_base=args.api_base,
        model=args.model,
        api_key=api_key,
        out_jsonl=out_jsonl,
        max_concurrency=args.max_concurrency,
    ))

    print(f"[Info] Done. Results saved to: {out_jsonl.resolve()}")


if __name__ == "__main__":
    main()
