"""C4 批量多轮修码 rollout 引擎（vLLM + CodeExecEnv）—— m3_rollout 的姊妹篇

与 m3 的唯一结构差异：工具调用从 <search>(检索) 换成 <run_code>(沙盒执行)。
训练层零改动即可复用（这正是本项目要证明的"环境可插拔"论点本身）。

锁步循环：每轮对活跃序列生成 → 解析 <run_code>/<answer> →
  run   → 沙盒执行(代码+可见测试) → 反馈文本作为 user 轮 → 继续
  answer→ 提取代码 → 查重(vs 参考解归一化) → grade_solution(隐藏测试) → 收尾
直到 </answer> 或 MAX_TURNS / MAX_RUNS 超限（超限交给格式奖励惩罚）。

M0 冒烟用法（docker 内，单卡）：
  docker run --rm --gpus '"device=0"' --security-opt seccomp=unconfined \
    -v /mnt/nas3/shared/model:/models:ro \
    -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
    -e HF_HUB_OFFLINE=1 1cat-vllm:v100-1.3.0 \
    python c4_rollout.py --model /models/Qwen3-4B-Instruct-2507 --n 8 --G 1
"""
import argparse
import json
import os

from vllm import LLM, SamplingParams

from c4_env import (SYSTEM, MAX_RUNS, MAX_TURNS, CodeExecEnv,
                    parse_trajectory_code, extract_code, grade_solution,
                    compute_reward_code)
from c4_tasks import norm_ws

MAX_NEW_TOKENS = 768   # 代码轮比检索轮长得多（m3 为 200/512）

USER_TMPL = """Task: {prompt}

Buggy implementation:
```python
{buggy_code}
```

Visible test:
{visible_test}

Its failure output:
{fail_output}

Fix the function. You can run code up to {max_runs} times (the sandbox has no internet and cannot write files). The final grade uses hidden tests, not the visible one."""


def load_tasks(tasks_path="data/c4/tasks.jsonl", hidden_path="data/c4/hidden.jsonl", n=None):
    tasks = [json.loads(l) for l in open(tasks_path)]
    hidden = {json.loads(l)["task_id"]: json.loads(l) for l in open(hidden_path)}
    if n:
        tasks = tasks[:n]
    return tasks, hidden


def rollout(model_path, tasks, hidden, G=1, max_seqs=128, temperature=1.0,
            dtype="float16", gpu_util=0.7, max_model_len=8192,
            max_new_tokens=MAX_NEW_TOKENS, llm=None):
    """返回每条轨迹（含 messages / turn_texts / grade / reward）。
    llm 可复用已加载实例（评测/训练交替时避免反复加载）。"""
    if llm is None:
        llm = LLM(model=model_path, dtype=dtype, gpu_memory_utilization=gpu_util,
                  max_model_len=max_model_len, enforce_eager=True)
    tok = llm.get_tokenizer()
    env = CodeExecEnv()
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                        top_p=1.0, stop=["</run_code>", "</answer>"])

    seqs = []
    for qi, t in enumerate(tasks):
        for _ in range(G):
            user = USER_TMPL.format(prompt=t["prompt"], buggy_code=t["buggy_code"],
                                    visible_test=t["visible_test"],
                                    fail_output=t["fail_output"], max_runs=MAX_RUNS)
            seqs.append({
                "qi": qi,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user}],
                "turn_texts": [], "n_runs": 0, "done": False,
            })
    active = list(range(len(seqs)))

    for turn in range(MAX_TURNS):
        if not active:
            break
        prompts = [
            tok.apply_chat_template(seqs[i]["messages"], tokenize=False,
                                    add_generation_prompt=True)
            for i in active
        ]
        outs = []
        for b0 in range(0, len(prompts), max_seqs):
            outs += llm.generate(prompts[b0:b0 + max_seqs], sp)
        next_active = []
        for j, i in enumerate(active):
            raw = outs[j].outputs[0].text.strip()
            s = seqs[i]
            s["turn_texts"].append(raw)
            kind, content, _ = parse_trajectory_code([raw])[0]
            t = tasks[s["qi"]]
            if kind == "run" and s["n_runs"] < MAX_RUNS:
                code = content or ""
                feedback, info = env.run_visible(code, t["visible_test"])
                s["n_runs"] += 1
                # 规范化存储：补上被 vLLM stop 剥掉的闭合标签（与 m3 同纪律）
                s["messages"].append({"role": "assistant",
                                      "content": f"<run_code>{code}</run_code>"})
                s["messages"].append({"role": "user", "content": feedback})
                next_active.append(i)
            else:
                # answer / invalid / run 超限：收尾（超限由 r_format/r_over 惩罚）
                if kind == "run":
                    stored = f"<run_code>{content or ''}</run_code>"
                elif kind == "answer":
                    stored = f"<answer>{content}</answer>"
                else:
                    stored = raw
                s["messages"].append({"role": "assistant", "content": stored})
                s["done"] = True
        active = next_active

    # ---- grading + reward（隐藏测试只在这里出现，永不进 prompt） ----
    results = []
    for s in seqs:
        t = tasks[s["qi"]]
        h = hidden[t["task_id"]]
        parsed = parse_trajectory_code(s["turn_texts"])
        answer_code, grade_info = None, None
        if parsed and parsed[-1][0] == "answer":
            answer_code = extract_code(parsed[-1][1])
        if answer_code:
            grade_info = grade_solution(answer_code, h)
            grade_info["code_extracted"] = True
            # 防 hacking 门：与参考解归一化后逐字符相同 → 判 0（背题/泄漏兜底）
            if norm_ws(answer_code) == h["ref_norm"]:
                grade_info["dup_ref"] = True
                grade_info["pass_rate"] = 0.0
        else:
            grade_info = {"code_extracted": False}
        reward, rinfo = compute_reward_code(s["turn_texts"], grade_info)
        results.append({
            "qi": s["qi"], "task_id": t["task_id"],
            "messages": s["messages"], "turn_texts": s["turn_texts"],
            "n_runs": s["n_runs"], "answer_code": answer_code,
            "grade": grade_info, "reward": reward, **rinfo,
        })
    return results, llm


def fmt_trajectory(r, tasks):
    """人类可读的轨迹 dump（M0 验收物）"""
    t = tasks[r["qi"]]
    lines = [f"===== task {r['task_id']} | {t['prompt'][:80]} ====="]
    for m in r["messages"][1:]:   # 跳过 system
        tag = m["role"].upper()
        lines.append(f"\n--- {tag} ---\n{m['content']}")
    g = r["grade"] or {}
    lines.append(f"\n--- GRADE ---\npass_rate={g.get('pass_rate')} orig={g.get('orig_rate')} "
                 f"plus={g.get('plus_rate')} code_ok={g.get('code_ok')} "
                 f"timed_out={g.get('timed_out')} dup_ref={g.get('dup_ref', False)}")
    lines.append(f"--- REWARD --- total={r['reward']:.4f} "
                 f"(r_pass={r['r_pass']} r_format={r['r_format']} r_over={r['r_over']} "
                 f"n_runs={r['n_runs']})")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--tasks", default="data/c4/tasks.jsonl")
    ap.add_argument("--hidden", default="data/c4/hidden.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--G", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--gpu_util", type=float, default=0.7)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--show", type=int, default=0, help="打印第 N 条轨迹（-1 不打印）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks, hidden = load_tasks(args.tasks, args.hidden, args.n)
    results, _ = rollout(args.model, tasks, hidden, G=args.G,
                         temperature=args.temperature, gpu_util=args.gpu_util,
                         dtype=args.dtype, max_new_tokens=args.max_new_tokens)

    n_traj = len(results)
    submitted = sum(1 for r in results if r["answer_code"])
    avg_pass = sum((r["grade"] or {}).get("pass_rate") or 0 for r in results) / n_traj
    avg_reward = sum(r["reward"] for r in results) / n_traj
    fmt_ok = sum(r["r_format"] for r in results) / n_traj
    avg_runs = sum(r["n_runs"] for r in results) / n_traj
    dups = sum(1 for r in results if (r["grade"] or {}).get("dup_ref"))
    tos = sum(1 for r in results if (r["grade"] or {}).get("timed_out"))
    print(json.dumps({"model": args.model, "n_traj": n_traj,
                      "submit_rate": round(submitted / n_traj, 4),
                      "avg_pass_rate": round(avg_pass, 4),
                      "avg_reward": round(avg_reward, 4),
                      "format_ok_pct": round(fmt_ok * 100, 2),
                      "avg_n_runs": round(avg_runs, 2),
                      "dup_ref": dups, "grade_timeout": tos},
                     ensure_ascii=False, indent=1))

    if args.show >= 0 and results:
        print("\n" + fmt_trajectory(results[args.show], tasks))

    out = args.out or "logs/c4_rollout_m0.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"\nsaved {n_traj} trajectories to {out}")
