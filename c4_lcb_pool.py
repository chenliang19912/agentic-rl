"""C4 LCB held-out 扩池 · Step1: LiveCodeBench test6 → 函数式题池（host 可跑，纯标准库）

过滤与归一化（预注册见 LCB_HELDOUT.md §一）：
- starter_code 非空（class Solution 函数式），签名解析出函数名/参数个数
- 测试逐条可解析：input 按行 split 后每行 literal_eval（行数==参数个数）；
  期望 literal_eval；期望或参数含 float 的剔除（== 精确语义）
- 题面含图片标记（"![("）或过短的剔除

输出 data/lcb/pool.jsonl：{tid, title, question_content, fn_name, signature, tests:[{inputs, expected}]}
用法：python3 c4_lcb_pool.py --in data/lcb/test6.jsonl --out data/lcb/pool.jsonl
"""
import argparse
import json
import re
from ast import literal_eval

FLOAT_MARK = ("+", "-", "e")  # literal 判定后复查 float 用 ast，不靠字符串

# 统一头部导入（预注册 §二.1）：typing/竞赛常用库，防候选码 `List` NameError。
# refgen 验证、AST 变异、grading 全链路同一份，保证语义一致
LCB_PRELUDE = (
    "from typing import List, Dict, Tuple, Optional, Set, Any\n"
    "import math, collections, heapq, itertools, functools, bisect, re, string, copy\n"
)


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
    from c4_env import sandbox_exec
    r = sandbox_exec(build_check_script(code, fn_name, tests), timeout)
    m = re.search(r"GRADES_JSON:(\{.*\})", r["stdout"])
    if not m:
        return False, (0, len(tests))
    g = json.loads(m.group(1))
    return (g["ok"] and g["pass"] == g["n"] and g["n"] == len(tests)), (g["pass"], g["n"])


def parse_signature(starter):
    """class Solution 风格签名 → (fn_name, [param names], typed_params_src)"""
    m = re.search(r"def\s+(\w+)\s*\(\s*self\s*,?\s*(.*?)\)\s*(->\s*[^:]+)?:", starter, re.S)
    if not m:
        return None, [], ""
    fn = m.group(1)
    params_src = m.group(2).strip()
    typed = m.group(3) or ""
    if not params_src:
        return None, [], ""
    names = []
    for p in params_src.split(","):
        name = p.split(":")[0].strip()
        if not name:
            return None, [], ""
        names.append(name)
    return fn, names, f"def {fn}({params_src}){typed}:"


def parse_one_test(t, n_params):
    """LCB functional 测试 → (inputs list, expected)；解析失败/含浮点返回 None"""
    lines = [l for l in t["input"].strip().split("\n")]
    if n_params == 1 and len(lines) != 1:
        # 单参数多行：整段当一行字面量（如多行 list）
        lines = [t["input"].strip()]
    if len(lines) != n_params:
        return None
    try:
        inputs = [literal_eval(l) for l in lines]
        expected = literal_eval(t["output"].strip())
    except Exception:
        return None

    def has_float(x):
        if isinstance(x, float):
            return True
        if isinstance(x, (list, tuple)):
            return any(has_float(i) for i in x)
        return False

    if has_float(expected) or any(has_float(i) for i in inputs):
        return None
    return inputs, expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/lcb/test6.jsonl")
    ap.add_argument("--out", default="data/lcb/pool.jsonl")
    args = ap.parse_args()

    stats = {"total": 0, "fn_style": 0, "no_sig": 0, "img_or_short": 0,
             "test_parse_fail": 0, "kept": 0}
    out = []
    for line in open(args.inp):
        x = json.loads(line)
        stats["total"] += 1
        starter = (x.get("starter_code") or "").strip()
        if not starter:
            continue
        stats["fn_style"] += 1
        fn, names, sig = parse_signature(starter)
        if not fn:
            stats["no_sig"] += 1
            continue
        meta = x.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        meta_fn = (meta or {}).get("func_name")
        if meta_fn and meta_fn != fn:      # 签名与 metadata 冲突 → 弃
            stats["no_sig"] += 1
            continue
        content = (x.get("question_content") or "").strip()
        if "![" in content or len(content) < 60:
            stats["img_or_short"] += 1
            continue
        def _load_tests(v):
            if isinstance(v, str):
                v = v.strip()
                if not v:
                    return []
                if v.startswith("eJ"):   # LCB 私有测试是 zlib+base64（json 或 pickle）
                    import base64, zlib
                    raw = zlib.decompress(base64.b64decode(v))
                    try:
                        return json.loads(raw.decode("utf-8"))
                    except UnicodeDecodeError:
                        import io, pickle
                        class _Restricted(pickle.Unpickler):
                            def find_class(self, module, name):
                                if module == "builtins":
                                    return getattr(__builtins__, name) if hasattr(__builtins__, name) else __builtins__[name]
                                raise pickle.UnpicklingError(f"forbidden: {module}.{name}")
                        obj = _Restricted(io.BytesIO(raw)).load()
                        # pickle 载荷通常是一段 JSON 文本（SHORT_BINUNICODE）
                        return json.loads(obj) if isinstance(obj, str) else obj
                return json.loads(v)
            return v or []

        pub = _load_tests(x["public_test_cases"])
        priv = _load_tests(x["private_test_cases"])
        tests, ok = [], True
        for t in list(pub) + list(priv):
            if t.get("testtype", "functional") != "functional":
                ok = False
                break
            r = parse_one_test(t, len(names))
            if r is None:
                ok = False
                break
            inputs, expected = r
            tests.append({"inputs": inputs, "expected": expected})
        if not ok or not priv:
            stats["test_parse_fail"] += 1
            continue
        tid = f"c4_lcb_{x['question_id']}_{fn}"
        out.append({"tid": tid, "title": x["question_title"],
                    "question_content": content, "fn_name": fn,
                    "signature": sig, "n_params": len(names),
                    "difficulty": x.get("difficulty", "?"),
                    "contest_date": x.get("contest_date", ""),
                    "tests": tests})
        stats["kept"] += 1

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["path"] = args.out
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
