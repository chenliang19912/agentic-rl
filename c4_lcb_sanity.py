"""C4 LCB 构建自检：refs 应全过（pass_rate=1.0），mutant 应不满分（构建语义成立）
用法（docker 内）：python c4_lcb_sanity.py"""
import json

from c4_env import grade_solution

tasks = [json.loads(l) for l in open("data/lcb/tasks.jsonl")]
hidden = {json.loads(l)["task_id"]: json.loads(l) for l in open("data/lcb/hidden.jsonl")}

bad = 0
for t in tasks[:8]:
    h = hidden[t["task_id"]]
    g_ref = grade_solution(h["ref_code"], h)
    g_mut = grade_solution(t["buggy_code"], h)
    ok = g_ref["pass_rate"] == 1.0 and g_mut["pass_rate"] < 1.0
    if not ok:
        bad += 1
    print(f"{t['task_id']}: ref={g_ref['pass_rate']:.3f} mut={g_mut['pass_rate']:.3f} "
          f"n_plus={g_ref['n_plus']} -> {'OK' if ok else 'BAD'}")
print(json.dumps({"checked": min(8, len(tasks)), "bad": bad}, ensure_ascii=False))
