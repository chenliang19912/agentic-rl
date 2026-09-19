"""C4 LCB held-out · Step3: 参考解 AST 单点变异 → tasks/hidden（docker 内跑）

完全复用 c4_tasks 的变异与验证管线（预注册 §二.2）：make_mutants(seed=42) →
可见测试必须失败 → 隐藏测试通过率 <0.95 → 沙盒生成失败输出。
差异仅在输入源：参考解来自基座生成（refs.jsonl），harness 由 LCB 测试直译。

用法（docker 内）：
  python c4_lcb_build.py --out data/lcb
输出 data/lcb/tasks.jsonl + hidden.jsonl（字段与 data/c4 完全同构）
"""
import argparse
import json
import random

from c4_env import sandbox_exec
from c4_tasks import make_mutants, _try_exec_visible, _plus_rate, norm_ws
from c4_lcb_pool import LCB_PRELUDE


def make_harness_prefix(task):
    # repr 而非 json.dumps：JSON 的 true/false 不是合法 Python 字面量。
    # 注意按 t['inputs']/t['expected'] 显式取——对 dict 直接二元解包会把键名
    # 字符串赋给变量（两键 dict 不报错、静默全错，M2 构建实测踩坑）
    tests_src = repr(task["tests"])
    return (
        LCB_PRELUDE
        + "inputs = []\n"
        + "results = []\n"
        + "for _t in " + tests_src + ":\n"
        + "    inputs.append(tuple(_t['inputs']))\n"
        + "    results.append(_t['expected'])\n"
        + "def assertion(got, exp, atol=0.0):\n"
        + "    assert got == exp, 'got=%r exp=%r' % (got, exp)\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/lcb/pool.jsonl")
    ap.add_argument("--refs", default="data/lcb/refs.jsonl")
    ap.add_argument("--out", default="data/lcb")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-plus-rate", type=float, default=0.95)
    args = ap.parse_args()

    pool = {json.loads(l)["tid"]: json.loads(l) for l in open(args.pool)}
    refs = {json.loads(l)["tid"]: json.loads(l) for l in open(args.refs)}
    rng = random.Random(args.seed)

    tasks, hidden = [], []
    stats = {"pool_with_ref": 0, "built": 0, "no_valid_mutant": 0,
             "mut_reject_visible_pass": 0, "mut_reject_plus_rate": 0,
             "by_difficulty": {}}
    for tid, ref_row in refs.items():
        if not ref_row.get("ref_code"):
            continue
        stats["pool_with_ref"] += 1
        p = pool[tid]
        ref_code = ref_row["ref_code"]
        fn = p["fn_name"]
        t0 = p["tests"][0]
        visible_test = (f"assert {fn}({', '.join(repr(a) for a in t0['inputs'])}) "
                        f"== {t0['expected']!r}")
        harness_prefix = make_harness_prefix(p)

        chosen = None
        for mut in make_mutants(ref_code, rng):
            ok, _ = _try_exec_visible(mut, visible_test)
            if ok is not False:
                stats["mut_reject_visible_pass"] += 1
                continue
            rate, n_plus = _plus_rate(mut, harness_prefix, fn, 0.0)
            if rate is None:
                rate, n_plus = 0.0, 0
            if not (0.0 <= rate < args.max_plus_rate):
                stats["mut_reject_plus_rate"] += 1
                continue
            chosen = (mut, rate, n_plus)
            break
        if chosen is None:
            stats["no_valid_mutant"] += 1
            print(f"[build] {tid} no_valid_mutant", flush=True)
            continue
        mut, mut_rate, n_plus = chosen

        r = sandbox_exec(
            mut + "\n\ntry:\n    " + visible_test +
            "\nexcept Exception:\n    import traceback; traceback.print_exc(limit=2)\n",
            timeout=5)
        fail_output = (r["stdout"] + r["stderr"]).strip()[-800:] or "(no output)"

        prompt = (p["question_content"]
                  + f"\n\n[Format] The function is a plain function (not wrapped in a class). "
                    f"Signature: {p['signature']} typing imports are already available.")
        tasks.append({
            "task_id": tid, "source": "lcb", "prompt": prompt,
            "func_name": fn, "buggy_code": mut, "visible_test": visible_test,
            "fail_output": fail_output, "n_plus": n_plus,
            "mut_plus_rate": round(mut_rate, 4),
            "difficulty": p["difficulty"],      # 难度标签，不进 prompt
        })
        hidden.append({
            "task_id": tid, "ref_code": ref_code, "fn_name": fn,
            "ref_norm": norm_ws(ref_code),
            "orig_tests": [], "harness_prefix": harness_prefix, "atol": 0.0,
            "plus_ok": True,
        })
        stats["built"] += 1
        d = stats["by_difficulty"].setdefault(p["difficulty"], 0)
        stats["by_difficulty"][p["difficulty"]] = d + 1
        print(f"[build] {tid} built={stats['built']} mut_rate={mut_rate:.2f}", flush=True)

    with open(f"{args.out}/tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(f"{args.out}/hidden.jsonl", "w") as f:
        for h in hidden:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    stats["out"] = args.out
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
