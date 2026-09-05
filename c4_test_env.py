"""C4 CPU 单测：沙盒防御层 / grading 正确性 / 奖励门控 / 任务生成器抽查 / 防泄漏审计

docker 内运行（无需 GPU；禁网用例需 seccomp=unconfined，缺失时该条标 SKIP-WARN）：
  docker run --rm --security-opt seccomp=unconfined \
    -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
    1cat-vllm:v100-1.3.0 python c4_test_env.py
"""
import json
import os
import sys

from c4_env import (sandbox_exec, CodeExecEnv, grade_solution,
                    parse_trajectory_code, extract_code, compute_reward_code,
                    _netns_available)

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name +
          ("" if cond else f"   <- {detail}"))


def skip(name, detail):
    SKIP.append(name)
    print(f"  SKIP-WARN {name}   ({detail})")


# ============ 1. 沙盒防御层 ============
print("[1] sandbox_exec 防御层")
r = sandbox_exec("print('hello-c4')", timeout=5)
check("基本执行 stdout/rc", "hello-c4" in r["stdout"] and r["rc"] == 0, r)

r = sandbox_exec("while True: pass", timeout=3)
check("死循环被超时杀掉", r["timed_out"] and r["rc"] != 0, r)

r = sandbox_exec("import socket; socket.create_connection(('1.1.1.1', 53), 2); print('NET_LEAK')", timeout=8)
if _netns_available():
    check("禁网生效（外连失败）", "NET_LEAK" not in r["stdout"] and r["rc"] != 0, r)
else:
    skip("禁网", "unshare -rn 不可用：docker 需 --security-opt seccomp=unconfined")

# 注意：必须显式 flush——临时文件对象 GC 时 flush 的 EFBIG 会被 Python 忽略（rc 仍 0），
# 初版测试因此误判"禁写失效"，实际写入已被 RLIMIT_FSIZE=0 拦截。
r = sandbox_exec("f = open('evil.txt', 'w')\nf.write('x')\nf.flush()\nprint('WRITE_OK')", timeout=5)
check("禁写文件（RLIMIT_FSIZE=0）", "WRITE_OK" not in r["stdout"] and r["rc"] != 0, r)

r = sandbox_exec("import os; print(sorted(os.environ))", timeout=5)
check("最小环境变量（无宿主 env 泄漏）", "AWS" not in r["stdout"] and "TOKEN" not in r["stdout"], r["stdout"][:200])

r = sandbox_exec("print('x' * 100000)", timeout=5)
check("输出截断 <= MAX_OUT_CHARS", len(r["stdout"]) <= 1200, len(r["stdout"]))

# ============ 2. run_visible（轮内工具调用） ============
print("[2] CodeExecEnv.run_visible")
env = CodeExecEnv()
fb, info = env.run_visible("def add(a, b):\n    return a + b", "assert add(2, 3) == 5")
check("正确代码 -> PASSED", info["passed"] and "VISIBLE_TEST: PASSED" in fb, fb)
fb, info = env.run_visible("def add(a, b):\n    return a - b", "assert add(2, 3) == 5")
check("错误代码 -> FAILED+traceback", not info["passed"] and "VISIBLE_TEST: FAILED" in fb
      and "AssertionError" in fb, fb)
fb, info = env.run_visible("def f():\n    while True: pass\nf()", "assert 1==1")
check("轮内死循环 -> timed_out 标记", info["timed_out"] and "timed out" in fb, fb)

# ============ 3. grade_solution（隐藏测试 grading） ============
print("[3] grade_solution")
HARNESS = ("def assertion(out, exp, atol):\n    assert out == exp, f'{out} != {exp}'\n"
           "inputs = [[1], [2], [3]]\nresults = [2, 4, 6]\n")
hidden_ok = {"orig_tests": ["assert double(1) == 2", "assert double(2) == 4"],
             "harness_prefix": HARNESS, "fn_name": "double", "atol": 0}
g = grade_solution("def double(x):\n    return x * 2", hidden_ok)
check("正确解 -> pass_rate=1.0", abs(g["pass_rate"] - 1.0) < 1e-9 and g["code_ok"], g)

g = grade_solution("def double(x):\n    return x + 1", hidden_ok)   # 只过 [1]->2
check("部分正确 -> 稠密分数 1/3", abs(g["pass_rate"] - 1 / 3) < 1e-6, g)

g = grade_solution("def double(x):\n    return x * 2", {**hidden_ok, "harness_prefix": ""})
check("无 harness -> 退化 orig 通过率", abs(g["pass_rate"] - 1.0) < 1e-9 and g["plus_rate"] is None, g)

g = grade_solution("while True: pass", hidden_ok)
check("死循环解 -> 超时且 0 分", g["timed_out"] and g["pass_rate"] == 0.0, g)

g = grade_solution("def double(:\n", hidden_ok)
check("语法错误解 -> code_ok=False 且 0 分", not g["code_ok"] and g["pass_rate"] == 0.0, g)

# ============ 4. 解析与奖励（含终止门控） ============
print("[4] parse / reward")
turns_good = ["<run_code>def f(): return 1</run_code>",
              "<answer>```python\ndef f():\n    return 2\n```</answer>"]
parsed = parse_trajectory_code(turns_good)
check("解析 run+answer", [k for k, _, _ in parsed] == ["run", "answer"], parsed)
check("extract_code 剥围栏", extract_code(parsed[1][1]) == "def f():\n    return 2",
      extract_code(parsed[1][1]))

grade_info = {"pass_rate": 0.8, "code_extracted": True, "plus_rate": 0.8, "orig_rate": 1.0}
total, info = compute_reward_code(turns_good, grade_info)
check("好轨迹 R=0.8+0.2", abs(total - 1.0) < 1e-9 and info["r_format"] == 1.0, info)

total, info = compute_reward_code(turns_good[:1], None)   # 修而不交
check("终止门控：无 answer -> r_pass=0", info["r_pass"] == 0.0 and info["r_format"] == 0.0, info)

turns_over = ["<run_code>x=1</run_code>"] * 7 + [turns_good[1]]
total, info = compute_reward_code(turns_over, grade_info)
check("超 run 上限 -> r_over>0 扣分", info["r_over"] > 0 and info["n_runs"] == 7, info)

turns_bad = ["I will fix it now", "<answer>```python\ndef f(): return 2\n```</answer>"]
total, info = compute_reward_code(turns_bad, grade_info)
check("invalid 轮 -> r_format=0", info["r_format"] == 0.0, info)

# ============ 5. 任务生成器抽查 + 防泄漏审计 ============
print("[5] c4_tasks 抽查")
tdir = "data/c4"
need_build = not os.path.exists(f"{tdir}/tasks.jsonl") or \
    os.path.getsize(f"{tdir}/tasks.jsonl") == 0
if need_build:
    print("  data/c4 缺失/为空，先小批量生成 (n=8)...")
    from c4_tasks import build
    build(8, seed=42, out_dir=tdir)

tasks = [json.loads(l) for l in open(f"{tdir}/tasks.jsonl")]
hiddens = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{tdir}/hidden.jsonl")}
check("任务数 > 0 且与 hidden 一一对应", len(tasks) > 0 and all(t["task_id"] in hiddens for t in tasks),
      f"{len(tasks)} tasks")

t = tasks[0]
h = hiddens[t["task_id"]]
fb, info = env.run_visible(t["buggy_code"], t["visible_test"])
check("首任务变异体确实失败", not info["passed"], fb[:200])
g = grade_solution(h["ref_code"], {**h, "fn_name": h["fn_name"]})
check("首任务参考解 grading 满分", abs(g["pass_rate"] - 1.0) < 1e-9, g)
g_mut = grade_solution(t["buggy_code"], h)
check("首任务变异体 grading < 1", g_mut["pass_rate"] < 1.0, g_mut)

leak = 0
for t in tasks:
    blob = json.dumps(t, ensure_ascii=False)
    h = hiddens[t["task_id"]]
    # 空 harness（plus_ok=False 的退化任务）不参与子串审计，防 "" in blob 恒真误报
    if (h["ref_code"] and h["ref_code"] in blob) or \
       (h["harness_prefix"] and h["harness_prefix"][:80] in blob):
        leak += 1
check("防泄漏审计：tasks.jsonl 不含参考解/harness", leak == 0, f"{leak} leaks")

# ============ 汇总 ============
print(f"\n==== PASS {len(PASS)} / FAIL {len(FAIL)} / SKIP {len(SKIP)} ====")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
