"""C4 LCB held-out · 参考解生成（单进程版，已被 c4_lcb_gen.py + c4_lcb_refs.py
两阶段方案取代——候选落盘 + 并行验证 + 早停。保留作历史记录。）
"""
import argparse
import ast
import json
import re

from vllm import LLM, SamplingParams

from c4_env import sandbox_exec
from c4_lcb_pool import LCB_PRELUDE, build_check_script, run_check

FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def normalize_candidate(code, fn_name):
    """LeetCode 习惯的 class Solution 包裹 → 模块级函数（提升全部方法、丢 self）。
    已是裸函数/无 Solution 类则原样返回。返回 (code, class_stripped)。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            methods = [sub for sub in node.body if isinstance(sub, ast.FunctionDef)]
            if not methods:
                return code, False
            for sub in methods:
                a = sub.args
                if a.args and a.args[0].arg == "self":
                    del a.args[0]
            new_tree = ast.Module(body=methods, type_ignores=[])
            ast.fix_missing_locations(new_tree)
            return ast.unparse(new_tree), True
    return code, False


def normalize_candidate(code, fn_name):
    """LeetCode 习惯的 class Solution 包裹 → 模块级函数（提升全部方法、丢 self）。
    已是裸函数/无 Solution 类则原样返回。返回 (code, class_stripped)。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            methods = [sub for sub in node.body if isinstance(sub, ast.FunctionDef)]
            if not methods:
                return code, False
            for sub in methods:
                a = sub.args
                if a.args and a.args[0].arg == "self":
                    del a.args[0]
            new_tree = ast.Module(body=methods, type_ignores=[])
            ast.fix_missing_locations(new_tree)
            return ast.unparse(new_tree), True
    return code, False


def build_check_script(code, fn_name, tests):
    """单候选全量测试脚本：逐 case 跑 fn(*inputs)==expected，GRADES_JSON 输出计数"""
    return (
        LCB_PRELUDE
        + "import json\n"
        + "_ns = {'__name__': '__main__'}\n"
        + "_ok = True\n"
        + "try:\n    exec(" + json.dumps(code) + ", _ns)\n"
        + "except Exception:\n    _ok = False\n"
        + "_pass = _n = 0\n"
        + "_fn = _ns.get(" + json.dumps(fn_name) + ")\n"
        + "_tests = " + repr(tests) + "\n"
        + "for t in _tests:\n"
        + "    _n += 1\n"
        + "    if not _ok or _fn is None:\n        continue\n"
        + "    try:\n"
        + "        if _fn(*t['inputs']) == t['expected']:\n            _pass += 1\n"
        + "    except (MemoryError, RecursionError, OverflowError):\n        raise\n"
        + "    except Exception:\n        pass\n"
        + "print('GRADES_JSON:' + json.dumps({'pass': _pass, 'n': _n, 'ok': _ok}))\n"
    )


def run_check(code, fn_name, tests, timeout=20):
    """候选码 + 全量测试 → (all_pass, (pass, n))；沙盒内执行"""
    r = sandbox_exec(build_check_script(code, fn_name, tests), timeout)
    m = re.search(r"GRADES_JSON:(\{.*\})", r["stdout"])
    if not m:
        return False, (0, len(tests))
    g = json.loads(m.group(1))
    return (g["ok"] and g["pass"] == g["n"] and g["n"] == len(tests)), (g["pass"], g["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--pool", default="data/lcb/pool.jsonl")
    ap.add_argument("--out", default="data/lcb/refs.jsonl")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool)]
    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=0.7,
              max_model_len=8192, enforce_eager=True)
    tok = llm.get_tokenizer()

    prompts, meta = [], []
    for p in pool:
        user = (
            p["question_content"]
            + f"\n\n[Format] Implement your solution as a plain Python function "
              f"(do NOT wrap it in a class):\n{p['signature']}\n"
              f"typing imports and common stdlib modules are already available. "
              f"Return only the final code in a ```python fence."
        )
        msgs = [{"role": "system",
                 "content": "You are an expert competitive programmer. Write correct, efficient Python."},
                {"role": "user", "content": user}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
        meta.append(p)

    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, seed=42)
    outs = llm.generate(prompts, sp)

    stats = {"pool": len(pool), "ref_found": 0, "no_ref": 0,
             "pass_any": 0, "class_stripped": 0, "by_difficulty": {}}
    rows = []
    for p, o in zip(meta, outs):
        n_pass = 0
        ref_code = None
        n_class = 0
        for cand in o.outputs:
            text = cand.text.strip()
            m = FENCE_RE.search(text)
            code = (m.group(1) if m else text).strip()
            if not code:
                continue
            code, stripped = normalize_candidate(code, p["fn_name"])
            n_class += int(stripped)
            blob = LCB_PRELUDE + "\n" + code
            all_pass, (cp, cn) = run_check(blob, p["fn_name"], p["tests"], timeout=30)
            if cp == cn and cn == len(p["tests"]):
                n_pass += 1
                if ref_code is None and all_pass:
                    ref_code = blob
        row = {"tid": p["tid"], "fn_name": p["fn_name"], "difficulty": p["difficulty"],
               "n_cand": len(o.outputs), "n_pass": n_pass, "n_class_stripped": n_class,
               "ref_code": ref_code}
        rows.append(row)
        d = stats["by_difficulty"].setdefault(p["difficulty"], {"n": 0, "ref": 0})
        d["n"] += 1
        if ref_code:
            stats["ref_found"] += 1
            d["ref"] += 1
        else:
            stats["no_ref"] += 1
        if n_pass:
            stats["pass_any"] += 1
        if n_class:
            stats["class_stripped"] += 1
        print(f"[refgen] {p['tid']} pass_any={n_pass}/{len(o.outputs)} "
              f"class={n_class} ref={'Y' if ref_code else 'N'}", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
