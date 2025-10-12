import json, re, random, argparse
from pathlib import Path
from typing import List, Tuple, Optional
from datasets import Dataset, DatasetDict  # pip install datasets

USER_BLOCK_RE = re.compile(
    r"<start_of_turn>\s*user\s*\n(.*?)<end_of_turn>",
    flags=re.DOTALL | re.IGNORECASE,
)

CTRL_TOKENS_RE = re.compile(
    r"</?s>|<\|.*?\|>|<bos>|<eos>|<pad>|<start_of_turn>|<end_of_turn>",
    flags=re.IGNORECASE,
)

def extract_user_prompt(raw: str) -> str:
    """尽量从 <start_of_turn>user ... <end_of_turn> 抽取；否则做一次清洗。"""
    m = USER_BLOCK_RE.search(raw)
    if m:
        content = m.group(1)
    else:
        content = CTRL_TOKENS_RE.sub("", raw)
    # 常见前缀“user\n”清理
    content = re.sub(r"^\s*user\s*\n", "", content, flags=re.IGNORECASE)
    return content.strip()

def pick_pair(responses: List[str], scores: List[float]) -> Optional[Tuple[int, int]]:
    """返回 (idx_chosen, idx_rejected)。优先 [最高分 vs 最低分]；若文本相同，尝试次低分。"""
    if not responses or not scores or len(responses) != len(scores):
        return None
    # 过滤空白回复
    valid = [(i, r.strip(), scores[i]) for i, r in enumerate(responses) if r and r.strip()]
    if len(valid) < 2:
        return None
    # 选极值
    valid_sorted = sorted(valid, key=lambda x: x[2])  # 按分数升序
    idx_lo, r_lo, s_lo = valid_sorted[0]
    idx_hi, r_hi, s_hi = valid_sorted[-1]
    if r_hi != r_lo:
        return idx_hi, idx_lo
    # 若最高与最低文本相同，尝试次低
    if len(valid_sorted) >= 3:
        idx_lo2, r_lo2, s_lo2 = valid_sorted[1]
        if r_lo2 != r_hi:
            return idx_hi, idx_lo2
    # 兜底：从不同文本里任选一个做负例
    for i, r, s in valid_sorted:
        if i != idx_hi and r != r_hi:
            return idx_hi, i
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="输入：你的原始 JSON（数组）")
    ap.add_argument("--out_jsonl", required=True, help="输出：DPO 格式 JSONL")
    ap.add_argument("--out_dir", required=True, help="输出：DPO 格式 hf dataset")
    ap.add_argument("--do_split", action="store_true", help="是否同时切分 train/test")
    ap.add_argument("--test_ratio", type=float, default=0.10, help="test 占比，默认 0.10")
    ap.add_argument("--seed", type=int, default=42, help="切分随机种子")
    ap.add_argument("--hf_format", action="store_true", help="选择输出hf格式的数据集还是jsonl")
    args = ap.parse_args()

    data = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
    out = []

    dropped_bad_shape = 0
    dropped_no_pair = 0

    for ex in data:
        try:
            raw_prompt = ex.get("prompt", "")
            user_prompt = extract_user_prompt(raw_prompt)
            responses = ex.get("all_generated_responses", []) or []
            scores = ex.get("all_rm_scores", []) or []
            pair = pick_pair(responses, scores)
            if not user_prompt or not pair:
                dropped_no_pair += 1
                continue
            i_ch, i_rj = pair
            ch_txt, rj_txt = responses[i_ch].strip(), responses[i_rj].strip()
            ch_sc, rj_sc = float(scores[i_ch]), float(scores[i_rj])

            out.append({
                "prompt":   [{"role": "user", "content": user_prompt}],
                "chosen":   [{"role": "assistant", "content": ch_txt}],
                "rejected": [{"role": "assistant", "content": rj_txt}],
                "score_chosen": ch_sc,
                "score_rejected": rj_sc,
            })
        except Exception:
            dropped_bad_shape += 1

    # 写出
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.do_split:
        with out_path.open("w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"✅ 转换完成：有效 {len(out)} 条，丢弃(无可用pair) {dropped_no_pair}，丢弃(异常) {dropped_bad_shape}")
        print(f"👉 已写入：{out_path}")
        return

    # 切分
    random.Random(args.seed).shuffle(out)
    n_total = len(out)
    n_test = max(1, int(round(n_total * args.test_ratio)))
    test_set = out[:n_test]
    train_set = out[n_test:]

    if args.hf_format:
        ds_train = Dataset.from_list(train_set)
        ds_test = Dataset.from_list(test_set)
        dsd = DatasetDict({"train": ds_train, "test": ds_test})

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dsd.save_to_disk(str(out_dir))
        print(f"👉 已保存到：{out_dir}")
    else:
        train_p = out_path.with_suffix(".train.jsonl")
        test_p  = out_path.with_suffix(".test.jsonl")
        with train_p.open("w", encoding="utf-8") as f:
            for r in train_set:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with test_p.open("w", encoding="utf-8") as f:
            for r in test_set:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"👉 训练集：{train_p}")
        print(f"👉 测试集：{test_p}")

    print(f"✅ 转换完成：总 {n_total} 条 | 训练 {len(train_set)} 条 | 测试 {len(test_set)} 条")

    print(f"（丢弃：无可用pair {dropped_no_pair}；异常 {dropped_bad_shape}）")

if __name__ == "__main__":
    main()
