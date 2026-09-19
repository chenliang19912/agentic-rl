"""C4 LCB held-out · 结果汇总（构建/评测完成后在 docker 或宿主机跑）
输出各 checkpoint 的 avg_pass / submit / full / avg_runs 与难度拆分"""
import glob
import json
from collections import defaultdict

tasks = {json.loads(l)["task_id"]: json.loads(l)
         for l in open("data/lcb/tasks.jsonl")}

rows = []
for path in sorted(glob.glob("logs/c4_lcb_*_multiturn.json")) + \
           sorted(glob.glob("logs/c4_lcb_base_oneshot.json")):
    name = path.split("/")[-1].replace("c4_lcb_", "").replace(".json", "")
    d = json.load(open(path))
    rs = d if isinstance(d, list) else d.get("results", [])
    if not rs:
        continue
    n = len(rs)
    ap = sum((r.get("grade") or {}).get("pass_rate") or 0 for r in rs) / n
    sub = sum(1 for r in rs if r.get("answer_code")) / n
    full = sum(1 for r in rs if (r.get("grade") or {}).get("pass_rate") == 1.0)
    runs = sum(r.get("n_runs", 0) for r in rs) / n
    dups = sum(1 for r in rs if (r.get("grade") or {}).get("dup_ref"))
    by = defaultdict(list)
    for r in rs:
        diff = tasks.get(r["task_id"], {}).get("difficulty", "?")
        by[diff].append((r.get("grade") or {}).get("pass_rate") or 0)
    bys = "  ".join(f"{k}={sum(v)/len(v):.3f}(n={len(v)})" for k, v in sorted(by.items()))
    print(f"{name:28s} n={n} avg_pass={ap:.4f} submit={sub:.3f} "
          f"full={full}/{n} runs={runs:.2f} dup={dups} | {bys}")

# train89 对照锚点（历史数字，口径见 C4_README）：base直答0.2370 base多轮0.4648
# RFT 0.7295 GRPO贪心峰0.7516 GiGPO峰0.7561
