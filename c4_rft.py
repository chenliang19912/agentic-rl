"""C4 M1 基线②：RFT（拒绝采样 SFT）—— 兜底基线

流程：
1. 在背题筛选后的训练集（tasks_train.jsonl）上，temperature=1.0 每题采样 G 条
   多轮修码轨迹（复用 c4_rollout.rollout，与 RL 同一轨迹分布，保证对照公平）
2. 过滤正例：隐藏测试 pass_rate >= 阈值 且 r_format==1（合法 <answer> 终止）
   且非 dup_ref（背题轨迹不进 SFT 数据——防污染扩散）
3. 每题至多保留 --max_per_task 条（按 reward 排序，防单题主导）
4. 落盘 SFT 数据（messages 对话格式，与 m2_sft_train.py 兼容）：
   训练: python m2_sft_train.py data/c4/rft_sft.jsonl checkpoints/c4_rft_lora 4096 2 8 /models/Qwen3-4B-Instruct-2507
   合并: python m2_merge.py checkpoints/c4_rft_lora checkpoints/c4_rft_merged /models/Qwen3-4B-Instruct-2507
5. RFT 基线评测：c4_rollout.py --model checkpoints/c4_rft_merged（多轮、贪心）

用法（docker 单卡）：
  python c4_rft.py --model /models/Qwen3-4B-Instruct-2507 --G 8
"""
import argparse
import json
import os
from collections import defaultdict

from c4_rollout import rollout, load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--tasks", default="data/c4/tasks_train.jsonl")
    ap.add_argument("--hidden", default="data/c4/hidden_train.jsonl")
    ap.add_argument("--n", type=int, default=None, help="限制任务数（调试用）")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--pass_threshold", type=float, default=1.0)
    ap.add_argument("--max_per_task", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out_sft", default="data/c4/rft_sft.jsonl")
    ap.add_argument("--out_pool", default="logs/c4_rft_pool.json")
    args = ap.parse_args()

    tasks, hidden = load_tasks(args.tasks, args.hidden, args.n)
    print(f"sample: {len(tasks)} tasks x G={args.G} @ temp={args.temperature}", flush=True)
    results, _ = rollout(args.model, tasks, hidden, G=args.G,
                         temperature=args.temperature)

    # ---- 正例过滤 ----
    pos = []
    for r in results:
        g = r["grade"] or {}
        if (g.get("pass_rate", 0.0) >= args.pass_threshold
                and r["r_format"] == 1.0
                and not g.get("dup_ref")):
            pos.append(r)
    # 每题按 reward 排序，至多 max_per_task 条
    by_task = defaultdict(list)
    for r in pos:
        by_task[r["task_id"]].append(r)
    kept = []
    for tid, rs in by_task.items():
        rs.sort(key=lambda r: (-r["reward"], r["n_runs"]))
        kept.extend(rs[:args.max_per_task])

    os.makedirs(os.path.dirname(args.out_sft) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_pool) or ".", exist_ok=True)
    with open(args.out_sft, "w") as f:
        for r in kept:
            f.write(json.dumps({"task_id": r["task_id"], "messages": r["messages"]},
                               ensure_ascii=False) + "\n")
    # 全轨迹池落盘（行为分析/复盘用，含失败轨迹）
    slim = [{k: r[k] for k in ("task_id", "n_runs", "reward", "r_pass", "r_format",
                               "grade", "turn_texts")} for r in results]
    with open(args.out_pool, "w") as f:
        json.dump(slim, f, ensure_ascii=False)

    n_dup = sum(1 for r in results if (r["grade"] or {}).get("dup_ref"))
    n_sub = sum(1 for r in results if r["answer_code"])
    stats = {
        "n_tasks": len(tasks), "G": args.G, "n_traj": len(results),
        "submit_rate": round(n_sub / len(results), 4),
        "n_pos_raw": len(pos), "n_pos_kept": len(kept),
        "pos_rate": round(len(pos) / len(results), 4),
        "task_coverage": round(len(by_task) / len(tasks), 4),
        "dup_ref_in_pool": n_dup,
        "avg_n_runs_pos": round(sum(r["n_runs"] for r in kept) / max(1, len(kept)), 2),
        "out_sft": args.out_sft,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
