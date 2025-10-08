import json
import argparse
from pathlib import Path
from statistics import mean
import numpy as np

def load_records(path: Path):
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

def main():
    ap = argparse.ArgumentParser(description="Left-tail selection by per-object mean(all_rm_scores)")
    ap.add_argument("--path", required=True, help="Path to .json or .jsonl")
    ap.add_argument("--key", default="all_rm_scores", help="Key name (default: all_rm_scores)")
    ap.add_argument("--q", type=float, default=None,
                    help="Custom left-quantile in (0,1), e.g., 0.03 for leftmost 3%%")
    ap.add_argument("--mean_thresh", type=float, default=None,
                    help="Absolute mean threshold; pick samples with mean < this value")
    ap.add_argument("--and_all_scores_le_global_mean", action="store_true",
                    help="Additionally require all single scores <= global mean (computed from all scores)")
    args = ap.parse_args()

    records = load_records(Path(args.path))

    per_means = []
    all_le_flags = []  # 是否每个分数 <= 全局均值（若需要）
    all_scores = []

    # 先收集全局均值（若需要）
    for obj in records:
        scores = obj.get(args.key, [])
        if isinstance(scores, list):
            all_scores.extend(scores)
    global_mean = (sum(all_scores)/len(all_scores)) if all_scores else float("nan")

    # 计算每个对象的均值及 all<=global_mean
    for obj in records:
        scores = obj.get(args.key, [])
        if isinstance(scores, list) and len(scores) > 0:
            m = mean(scores)
            per_means.append(m)
            all_le = all(s <= global_mean for s in scores)
            all_le_flags.append(all_le)
        else:
            # 无分数的样本，设为 NaN，并标记 False
            per_means.append(float("nan"))
            all_le_flags.append(False)

    # 过滤掉 NaN
    per_means_arr = np.array(per_means, dtype=float)
    valid_mask = ~np.isnan(per_means_arr)
    means = per_means_arr[valid_mask]
    flags = np.array(all_le_flags, dtype=bool)[valid_mask]
    total = means.size

    if total == 0:
        print("没有有效样本（每个对象缺少分数）。")
        return

    print(f"[Info] 样本总数: {total}")
    print(f"[Info] 样本均值范围: min={means.min():.6f}, max={means.max():.6f}")
    print(f"[Info] 全局均值(用于可选条件 all_scores<=global_mean): {global_mean:.6f}")

    # 常用分位点报告
    default_qs = [0.005, 0.01, 0.02, 0.05, 0.10]
    print("\n=== 左侧尾部（默认常用分位点） ===")
    for q in default_qs:
        thr = np.quantile(means, q)
        mask_q = means < thr
        if args.and_all_scores_le_global_mean:
            mask_q = mask_q & flags
        cnt = int(mask_q.sum())
        pct = cnt / total * 100.0
        print(f"q={q:.3f}  阈值={thr:.6f}  ->  样本数={cnt}  占比={pct:.2f}%")

    # 自定义分位数
    if args.q is not None:
        if not (0.0 < args.q < 1.0):
            raise ValueError("--q 必须在 (0,1) 内，比如 0.03 表示最左 3%")
        thr = np.quantile(means, args.q)
        mask_q = means < thr
        if args.and_all_scores_le_global_mean:
            mask_q = mask_q & flags
        cnt = int(mask_q.sum())
        pct = cnt / total * 100.0
        print(f"\n=== 自定义分位数 ===")
        print(f"q={args.q:.4f}  阈值={thr:.6f}  ->  样本数={cnt}  占比={pct:.2f}%")

    # 绝对阈值
    if args.mean_thresh is not None:
        thr = args.mean_thresh
        mask_t = means < thr
        if args.and_all_scores_le_global_mean:
            mask_t = mask_t & flags
        cnt = int(mask_t.sum())
        pct = cnt / total * 100.0
        print(f"\n=== 绝对均值阈值 ===")
        print(f"mean < {thr:.6f}  ->  样本数={cnt}  占比={pct:.2f}%")

if __name__ == "__main__":
    main()
