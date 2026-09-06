"""C4 M1 基线①：基座直答（one-shot，无执行反馈）+ 背题筛选

口径（见 C4_README §四）：
- 单次生成：同任务 prompt，但声明"无代码执行"，要求直接 <answer> 提交修复代码
- 贪心 temperature=0（EvalPlus pass@1 greedy 口径）
- 判分与多轮 rollout 完全同一套（grade_solution + 查重门），保证三组数字可比

背题筛选（C4_README §3.4 对策）：
- 直答 pass_rate == 1.0 的题 → 标记 memorized_or_trivial（模型已会/背过，
  无训练信号且有污染风险）→ 从训练集剔除
- 产出 data/c4/tasks_train.jsonl / hidden_train.jsonl（剔除后）
- 剔除率本身是可报告的实证数字（MBPP 对 Qwen3 的污染率）

用法（docker 单卡）：
  python c4_baseline.py --model /models/Qwen3-4B-Instruct-2507
"""
import argparse
import json
import os

from vllm import LLM, SamplingParams

from c4_env import (extract_code, grade_solution, ANSWER_RE)
from c4_tasks import norm_ws
from c4_rollout import USER_TMPL, load_tasks
from c4_env import MAX_RUNS

SYSTEM_ONESHOT = (
    "You are a code-repair expert. You are given a buggy Python function and one failing visible test.\n"
    "You CANNOT execute code. Directly output the full fixed function, exactly in this format:\n"
    "<answer>```python\nyour full fixed function\n```</answer>"
)


def run_baseline(model_path, tasks, hidden, max_new_tokens=768,
                 dtype="float16", gpu_util=0.7, max_model_len=8192):
    llm = LLM(model=model_path, dtype=dtype, gpu_memory_utilization=gpu_util,
              max_model_len=max_model_len, enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0,
                        top_p=1.0, stop=["</answer>"])

    prompts = []
    for t in tasks:
        user = USER_TMPL.format(prompt=t["prompt"], buggy_code=t["buggy_code"],
                                visible_test=t["visible_test"],
                                fail_output=t["fail_output"], max_runs=MAX_RUNS) + \
            "\n\nYou cannot run code here. Submit the fixed function directly."
        msgs = [{"role": "system", "content": SYSTEM_ONESHOT},
                {"role": "user", "content": user}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
    outs = llm.generate(prompts, sp)

    results = []
    for t, o in zip(tasks, outs):
        raw = o.outputs[0].text.strip()
        h = hidden[t["task_id"]]
        m = ANSWER_RE.search(raw)
        code = extract_code(m.group(1).strip()) if m else None
        if code:
            g = grade_solution(code, h)
            g["code_extracted"] = True
            if norm_ws(code) == h["ref_norm"]:
                g["dup_ref"] = True
                g["pass_rate"] = 0.0
        else:
            g = {"code_extracted": False, "pass_rate": 0.0, "dup_ref": False,
                 "orig_rate": 0.0, "plus_rate": None, "timed_out": False}
        results.append({"task_id": t["task_id"], "source": t["source"],
                        "submitted": bool(code), "answer_code": code,
                        "raw": raw, "grade": g})
    return results, llm


def summarize_and_filter(results, tasks, hidden, model, out_json, filter_dir):
    """统计 + 落盘 + 背题筛选（抽成函数供修复脚本复用）"""
    n = len(results)
    submit = sum(r["submitted"] for r in results)
    # grade_solution 只在命中查重时才写 dup_ref 键，统一用 .get()
    solved = [r for r in results if r["grade"]["pass_rate"] == 1.0 and not r["grade"].get("dup_ref")]
    dup = [r for r in results if r["grade"].get("dup_ref")]
    avg_pass = sum(r["grade"]["pass_rate"] for r in results) / n
    partial = sum(1 for r in results if 0 < r["grade"]["pass_rate"] < 1)
    zero = sum(1 for r in results if r["grade"]["pass_rate"] == 0 and not r["grade"].get("dup_ref"))

    # 按源拆分污染率（MBPP+ vs HumanEval+ 对照）
    by_src = {}
    for r in results:
        s = by_src.setdefault(r["source"], {"n": 0, "solved": 0, "dup": 0})
        s["n"] += 1
        if r["grade"]["pass_rate"] == 1.0 and not r["grade"].get("dup_ref"):
            s["solved"] += 1
        if r["grade"].get("dup_ref"):
            s["dup"] += 1
    for s in by_src.values():
        s["exclusion_rate"] = round((s["solved"] + s["dup"]) / max(1, s["n"]), 4)

    summary = {
        "model": model, "n_tasks": n,
        "submit_rate": round(submit / n, 4),
        "avg_pass_rate": round(avg_pass, 4),
        "solved_full": len(solved),          # 直答满分（背题/过于简单候选）
        "dup_ref_verbatim": len(dup),        # 逐字背参考解（查重门判0）
        "partial": partial, "zero": zero,
        "train_keep": n - len(solved) - len(dup),
        "exclusion_rate": round((len(solved) + len(dup)) / n, 4),
        "by_source": by_src,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False)
    print(f"saved baseline to {out_json}")

    # ---- 背题筛选：直答满分 或 逐字 dup → 从训练集剔除 ----
    os.makedirs(filter_dir, exist_ok=True)   # 评测模式可能指向不存在的隔离目录
    drop = {r["task_id"] for r in solved} | {r["task_id"] for r in dup}
    keep_t = [t for t in tasks if t["task_id"] not in drop]
    keep_h = [hidden[t["task_id"]] for t in keep_t]
    with open(os.path.join(filter_dir, "tasks_train.jsonl"), "w") as f:
        for t in keep_t:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(os.path.join(filter_dir, "hidden_train.jsonl"), "w") as f:
        for h in keep_h:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    with open(os.path.join(filter_dir, "excluded.jsonl"), "w") as f:
        for r in results:
            if r["task_id"] in drop:
                f.write(json.dumps({"task_id": r["task_id"],
                                    "reason": "dup_ref_verbatim" if r["grade"].get("dup_ref") else "solved_oneshot",
                                    "pass_rate": r["grade"]["pass_rate"]},
                                   ensure_ascii=False) + "\n")
    print(f"train set: {len(keep_t)}/{n} kept -> {filter_dir}/tasks_train.jsonl")
    print(f"excluded {len(drop)} tasks -> {filter_dir}/excluded.jsonl")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--tasks", default="data/c4/tasks.jsonl")
    ap.add_argument("--hidden", default="data/c4/hidden.jsonl")
    ap.add_argument("--out", default="logs/c4_baseline_oneshot.json")
    ap.add_argument("--filter_out", default="data/c4", help="筛选后训练集目录")
    args = ap.parse_args()

    tasks, hidden = load_tasks(args.tasks, args.hidden)
    results, _ = run_baseline(args.model, tasks, hidden)
    summarize_and_filter(results, tasks, hidden, args.model, args.out, args.filter_out)
