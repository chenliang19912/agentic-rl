"""修复 4 个重复 HE task_id（HumanEval 不同题共用函数名：add/sum_squares/solve/
correct_bracketing 各×2）。

撞 id 的后果：hidden 按 dict 加载 → 后行覆盖前行 → 这 8 行任务全部用了
错误的参考解/测试判分（M1 全池基线实测踩坑，keep 89+excluded 286=375≠378 暴露）。

修复步骤：
1. 重复 tid 重命名为 {tid}_a / {tid}_b（两行是不同 HE 题，都是有效任务，全保留）
2. 重写 tasks.jsonl / hidden.jsonl（c4_tasks.py 根因已修：tid 带 HE 题号）
3. GPU 补跑这 8 题的直答基线（旧 grade 不可信）
4. 合并进 logs/c4_baseline_oneshot.json（替换旧 8 行）→ 重算 summary + 筛选

用法：docker 内（GPU）python -u c4_fix_dup.py
"""
import json
from collections import Counter

from c4_baseline import run_baseline, summarize_and_filter
from c4_rollout import load_tasks

TASKS, HIDDEN = "data/c4/tasks.jsonl", "data/c4/hidden.jsonl"
OUT_JSON, FILTER_DIR = "logs/c4_baseline_oneshot.json", "data/c4"
MODEL = "/models/Qwen3-4B-Instruct-2507"

tasks = [json.loads(l) for l in open(TASKS)]
hidden_rows = [json.loads(l) for l in open(HIDDEN)]
assert len(tasks) == len(hidden_rows) and all(
    t["task_id"] == h["task_id"] for t, h in zip(tasks, hidden_rows)), "行序不一一对应"

# ---- 1) rename ----
c = Counter(t["task_id"] for t in tasks)
dups = {k for k, v in c.items() if v > 1}
print("dup ids:", sorted(dups))
seen = Counter()
fix_idx = []
for i, t in enumerate(tasks):
    tid = t["task_id"]
    if tid in dups:
        seen[tid] += 1
        new_tid = f"{tid}_a" if seen[tid] == 1 else f"{tid}_b"
        tasks[i]["task_id"] = new_tid
        hidden_rows[i]["task_id"] = new_tid
        fix_idx.append(i)
assert len(fix_idx) == 2 * len(dups), fix_idx
assert len({t["task_id"] for t in tasks}) == len(tasks), "仍有重复"

# ---- 2) 重写 ----
with open(TASKS, "w") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
with open(HIDDEN, "w") as f:
    for h in hidden_rows:
        f.write(json.dumps(h, ensure_ascii=False) + "\n")
print(f"rewrote {len(tasks)} tasks / hidden with unique ids")

# ---- 3) 补跑 8 题基线 ----
fix_tasks = [tasks[i] for i in fix_idx]
fix_hidden = {hidden_rows[i]["task_id"]: hidden_rows[i] for i in fix_idx}
print(f"re-running baseline on {len(fix_tasks)} fixed tasks ...", flush=True)
results8, _ = run_baseline(MODEL, fix_tasks, fix_hidden)

# ---- 4) 合并 + 重算 ----
old = json.load(open(OUT_JSON))
merged = [r for r in old["results"] if r["task_id"] not in dups] + results8
by_id = {r["task_id"]: r for r in merged}
ordered = [by_id[t["task_id"]] for t in tasks]   # 与 tasks 行序对齐
hidden_map = {h["task_id"]: h for h in hidden_rows}
summarize_and_filter(ordered, tasks, hidden_map, MODEL, OUT_JSON, FILTER_DIR)
print("FIX_DUP_DONE")
