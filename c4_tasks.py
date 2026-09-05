"""C4 任务生成器：EvalPlus(MBPP+ / HumanEval+) → AST 变异注入 bug → 验证 → 可见/隐藏字段分离

流程（M1 起双数据源扩池，背景见 C4_README §3.4）：
1. 载入 EvalPlus 数据集并归一化为统一记录（load_pool）：
   - MBPP+：code(参考解) / test_list(原始 assert) / test(plus harness，顶层 inputs/results)
   - HumanEval+：prompt(签名+docstring)+canonical_solution(函数体) 拼成参考解；
     entry_point 即被测函数名；test 的 check(candidate) 体内含 inputs/results，
     重建为 MBPP 同构的顶层 harness_prefix；可见测试从 check 案例拼装
     （优先非浮点 case，== 断言可靠）
2. 对参考解做单点 AST 变异（比较符/算术符/常量±1/布尔/and-or/删循环体语句）
3. 验证变异体：可见测试必须失败（prompt 里要展示失败输出）、plus 通过率 < 0.95、
   3 秒内可执行完；记录 plus 通过率作为难度标签
4. 落盘两份文件（防泄漏审计，检索 Agent 项目纪律移植）：
   - tasks.jsonl：模型可见字段（prompt/坏代码/可见测试/失败输出）
   - hidden.jsonl：grading 私有字段（参考解/原始测试/harness_prefix/atol/fn_name/ref_norm）
     —— rollout 进程只把 hidden 交给 grade_solution，永不进 prompt

用法：docker 内
  python c4_tasks.py --n 400 --seed 42 --out data/c4 \
    --parquets data/mbpp/mbppplus.parquet,data/mbpp/humanevalplus.parquet
"""
import argparse
import ast
import json
import os
import random
import re
import signal

import pandas as pd

from c4_env import sandbox_exec

PARQUET = "data/mbpp/mbppplus.parquet"
PARQUET_HE = "data/mbpp/humanevalplus.parquet"

# ---------------- AST 变异 ----------------
# 实现要点：变异按"前序遍历位点序号"定位（_MutCollector 计数 == _Mutator 计数），
# 每个变异体独立重新 parse + NodeTransformer 改写，保证变异一定落在被 unparse 的树上。
# （初版 bug 教训：site 携带旧树节点引用，改写旧树却 unparse 新树 → 变异体==原文）
CMP_SWAPS = {
    ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.Lt, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}
BIN_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
    ast.FloorDiv: ast.Div, ast.Mod: ast.FloorDiv, ast.BitAnd: ast.BitOr,
}


def _is_site_constant(node):
    if isinstance(node.value, bool):
        return True
    return isinstance(node.value, int) and -1000 <= node.value <= 1000


class _MutCollector(ast.NodeVisitor):
    """前序遍历计数所有可变异位点（与 _Mutator 的计数顺序严格一致）"""
    def __init__(self):
        self.n = 0

    def visit_Compare(self, node):
        for op in node.ops:
            if type(op) in CMP_SWAPS:
                self.n += 1
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if type(node.op) in BIN_SWAPS:
            self.n += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.n += 1
        self.generic_visit(node)

    def visit_Constant(self, node):
        if _is_site_constant(node):
            self.n += 1
        self.generic_visit(node)

    def visit_For(self, node):
        if len(node.body) >= 2:
            self.n += 1
        self.generic_visit(node)


class _Mutator(ast.NodeTransformer):
    """对前序遍历中第 target 个位点做单点变异"""
    def __init__(self, target, rng):
        self.target = target
        self.rng = rng
        self.count = 0
        self.done = False

    def _hit(self):
        if self.done:
            return False
        if self.count == self.target:
            self.done = True
            self.count += 1
            return True
        self.count += 1
        return False

    def visit_Compare(self, node):
        for i, op in enumerate(node.ops):
            if type(op) in CMP_SWAPS:
                if self._hit():
                    node.ops[i] = CMP_SWAPS[type(op)]()
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node):
        if type(node.op) in BIN_SWAPS:
            if self._hit():
                node.op = BIN_SWAPS[type(node.op)]()
        self.generic_visit(node)
        return node

    def visit_BoolOp(self, node):
        if self._hit():
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):
        if _is_site_constant(node):
            if self._hit():
                if isinstance(node.value, bool):
                    node.value = not node.value
                else:
                    node.value = node.value + (1 if self.rng.random() < 0.5 else -1)
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        if len(node.body) >= 2:
            if self._hit():
                del node.body[-1]
        self.generic_visit(node)
        return node


def make_mutants(ref_code, rng, k=6):
    """返回至多 k 个单点变异源码（未验证）；过滤掉变异后与原文相同的"""
    out = []
    try:
        base = ast.parse(ref_code)
        orig_src = ast.unparse(base)
    except SyntaxError:
        return out
    coll = _MutCollector()
    coll.visit(base)
    idxs = list(range(coll.n))
    rng.shuffle(idxs)
    for j in idxs[:k * 3]:
        if len(out) >= k:
            break
        try:
            tree = _Mutator(j, rng).visit(ast.parse(ref_code))
            ast.fix_missing_locations(tree)
            src = ast.unparse(tree)
        except Exception:
            continue
        if src and src != orig_src:
            out.append(src)
    return out


# ---------------- 变异体验证（guarded 子进程 + SIGALRM 双层防护） ----------------
# M1 构建卡死取证（c4_he_get_odd_collatz）：collatz 变异体整数指数爆炸，
# 单次 bigint 乘法是 C 级调用——SIGALRM 只在字节码边界处理，根本拦不住；
# 内存 10 分钟膨胀到 46.8GiB（前两次构建"静默死亡"= 膨胀到宿主机 OOM kill）。
# 对策：验证 exec 移入 fork 子进程，RLIMIT_CPU（SIGXCPU 是内核级杀，不依赖
# 字节码边界）+ RLIMIT_AS 4GB（bigint 分配失败即 MemoryError）+ 父进程超时 kill。
import multiprocessing as mp
import resource

MEM_LIMIT = 4 * 1024 ** 3
_MP_CTX = mp.get_context("fork")


class _Timeout(Exception):
    pass


def _alarm(sec):
    def h(signum, frame):
        raise _Timeout()
    signal.signal(signal.SIGALRM, h)
    signal.alarm(sec)


def _guarded(fn, sec):
    """在带 RLIMIT_CPU/AS 的 fork 子进程里执行 fn()。
    返回 ("ok", result) / ("exc", 类名, msg) / ("timeout",) / ("killed",)"""
    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)

    def worker():
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (sec, sec + 1))
            resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
            signal.alarm(0)
            out = fn()
            child_conn.send(("ok", out))
        except BaseException as e:
            try:
                child_conn.send(("exc", type(e).__name__, str(e)[:200]))
            except Exception:
                pass
        finally:
            child_conn.close()
            os._exit(0)

    p = _MP_CTX.Process(target=worker)
    p.start()
    child_conn.close()
    if parent_conn.poll(sec + 4):
        try:
            got = parent_conn.recv()
        except Exception:
            got = ("killed",)
    else:
        got = ("timeout",)
    p.kill()
    p.join()
    parent_conn.close()
    return got


def _try_exec_visible(code, visible_test, sec=3):
    """执行 code + visible assert；返回 (ok, err_text)。
    ok=True 断言通过 / False 断言失败或异常 / None 超时或跑飞（弃用该变异体）。"""
    def fn():
        ns = {"__name__": "__t__"}
        exec(code, ns)
        exec(visible_test, dict(ns))
        return True
    st = _guarded(fn, sec)
    if st[0] == "ok":
        return True, ""
    if st[0] == "exc":
        # 资源型异常 = 变异体跑飞（bigint 爆炸/无限递归），必须弃用而非当作
        # "断言失败"——否则失控变异体会混进任务集，grading/rollout 时被拖死
        if st[1] in ("MemoryError", "RecursionError", "OverflowError", "TimeoutError"):
            return None, st[1]
        return False, f"{st[1]}: {st[2]}"
    return None, st[0].upper()


def _plus_rate(code, harness_prefix, fn_name, atol, sec=10):
    """guarded 子进程里逐 case 跑 plus harness，返回 (rate, n) 或 (None, 0)"""
    if not harness_prefix:
        return None, 0

    def fn():
        ns = {"__name__": "__t__"}
        exec(code, ns)
        exec(harness_prefix, ns)
        f = ns[fn_name]
        assertion = ns["assertion"]
        inputs, results = ns["inputs"], ns["results"]
        npass = 0
        for inp, exp in zip(inputs, results):
            try:
                assertion(f(*inp), exp, atol)
                npass += 1
            except (MemoryError, RecursionError, OverflowError):
                raise   # 资源型异常=跑飞，必须上抛给 guard 判废（不能被逐 case 吞掉）
            except Exception:
                pass
        return npass / max(1, len(results)), len(results)

    st = _guarded(fn, sec)
    if st[0] == "ok":
        return st[1]
    return None, 0


# ---------------- harness 拆分（MBPP+） ----------------
LOOP_RE = re.compile(r"^for\s+.*:\s*$", re.M)
ATOL_RE = re.compile(r"assertion\([^()]*,\s*([0-9.eE+-]+)\s*\)\s*$", re.M)


def split_harness(harness_src):
    """MBPP+：截掉尾部顶层 for 循环 → harness_prefix；提取 atol。
    返回 (prefix, atol, ok)。ok=False 时 grading 退化为仅原始 assert。"""
    if not harness_src:
        return "", 0, False
    loops = list(LOOP_RE.finditer(harness_src))
    if not loops:
        return "", 0, False
    cut = loops[-1].start()
    prefix, tail = harness_src[:cut], harness_src[cut:]
    m = ATOL_RE.search(tail)
    try:
        atol = float(m.group(1)) if m else 0.0
    except ValueError:
        atol = 0.0
    ns = {"__name__": "__h__"}
    try:
        _alarm(5)
        exec(prefix, ns)
        signal.alarm(0)
        ok = all(k in ns for k in ("inputs", "results", "assertion"))
    except Exception:
        signal.alarm(0)
        ok = False
    finally:
        signal.alarm(0)
    return (prefix, atol, ok) if ok else ("", atol, False)


# ---------------- harness 重建（HumanEval+） ----------------
def split_humaneval_harness(test_src):
    """HumanEval+ test = is_floats/assertion 定义 + def check(candidate)（inputs/
    results/for 在 check 体内）。重建为 MBPP 同构的顶层 harness_prefix。
    返回 (prefix, atol=0.0, ok)。"""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return "", 0.0, False
    check_fn = next((n for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name == "check"), None)
    if check_fn is None:
        return "", 0.0, False
    lines = test_src.splitlines(keepends=True)
    defs_src = "".join(lines[:check_fn.lineno - 1])   # check 之前的定义部分
    assigns = {}
    for st in check_fn.body:
        if (isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and st.targets[0].id in ("inputs", "results")):
            seg = ast.get_source_segment(test_src, st)
            if seg:
                assigns[st.targets[0].id] = seg.strip()
    if len(assigns) != 2:
        return "", 0.0, False
    prefix = defs_src + "\n" + assigns["inputs"] + "\n" + assigns["results"] + "\n"
    ns = {"__name__": "__h__"}
    try:
        _alarm(10)
        exec(prefix, ns)
        signal.alarm(0)
        ok = all(k in ns for k in ("inputs", "results", "assertion"))
    except Exception:
        signal.alarm(0)
        ok = False
    finally:
        signal.alarm(0)
    return (prefix, 0.0, ok) if ok else ("", 0.0, False)


def _has_float(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return any(_has_float(i) for i in x)
    return False


def make_visible_from_harness(harness_prefix, fn_name):
    """HumanEval 无 test_list：从 harness 的 inputs/results 拼单行可见 assert。
    优先非浮点 case（== 断言无精度争议）；全浮点则取首个 case。"""
    ns = {"__name__": "__v__"}
    try:
        _alarm(10)
        exec(harness_prefix, ns)
        signal.alarm(0)
    except Exception:
        signal.alarm(0)
        return None
    finally:
        signal.alarm(0)
    inputs, results = ns.get("inputs"), ns.get("results")
    if not inputs or not results:
        return None
    pairs = list(zip(inputs, results))
    ordered = [p for p in pairs if not _has_float(p[0]) and not _has_float(p[1])] or pairs
    inp, exp = ordered[0]
    try:
        args = ", ".join(repr(a) for a in inp)
        return f"assert {fn_name}({args}) == {exp!r}"
    except Exception:
        return None


# ---------------- 数据池统一加载 ----------------
def entry_fn_name(code, test_list):
    """MBPP：从首个 assert 提取被测函数名；失败则取 code 的第一个顶层函数"""
    m = re.search(r"assert\s+(\w+)\s*\(", test_list[0]) if test_list else None
    if m:
        return m.group(1)
    try:
        for node in ast.parse(code).body:
            if isinstance(node, ast.FunctionDef):
                return node.name
    except SyntaxError:
        pass
    return None


def load_pool(parquets):
    """多个 EvalPlus parquet → 统一记录列表（字段见文件头 docstring）"""
    pool = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        if "canonical_solution" in df.columns:          # HumanEval+
            for _, row in df.iterrows():
                prompt = str(row["prompt"])
                ref = (prompt + str(row["canonical_solution"])).strip()
                fn = str(row["entry_point"])
                prefix, atol, ok = split_humaneval_harness(str(row["test"]))
                vt = make_visible_from_harness(prefix, fn) if ok else None
                if not ok or not vt:
                    continue
                # tid 必须带题号：HumanEval 存在不同题共用函数名（add/solve/
                # sum_squares/correct_bracketing 各×2），仅用 fn 会撞 id →
                # hidden dict 加载时后者覆盖前者 → 判分用错参考解（M1 实测踩坑）
                he_num = str(row["task_id"]).split("/")[-1]
                pool.append({"source": "humaneval+", "tid": f"c4_he_{fn}_{he_num}",
                             "prompt": prompt.strip(), "ref_code": ref,
                             "fn_name": fn, "orig_tests": [],
                             "harness_prefix": prefix, "atol": atol,
                             "plus_ok": True, "visible_test": vt})
        else:                                            # MBPP+
            for _, row in df.iterrows():
                ref = str(row["code"]).strip()
                tl = [str(t).strip() for t in row["test_list"]]
                if not tl:
                    continue
                fn = entry_fn_name(ref, tl)
                if not fn:
                    continue
                prefix, atol, ok = split_harness(str(row["test"]))
                pool.append({"source": "mbpp+", "tid": f"c4_mb_{int(row['task_id'])}",
                             "prompt": str(row["prompt"]).strip(), "ref_code": ref,
                             "fn_name": fn, "orig_tests": tl,
                             "harness_prefix": prefix if ok else "", "atol": atol,
                             "plus_ok": ok, "visible_test": tl[0]})
    return pool


# ---------------- 主构建 ----------------
def norm_ws(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def build(n_tasks, seed=42, out_dir="data/c4", parquets=(PARQUET, PARQUET_HE),
          max_plus_rate=0.95, min_plus_rate=0.0):
    rng = random.Random(seed)
    pool = load_pool(list(parquets))
    rng.shuffle(pool)

    os.makedirs(out_dir, exist_ok=True)
    tasks, hidden = [], []
    stats = {"pool": len(pool), "built": 0, "no_valid_mutant": 0,
             "mut_reject_visible_pass": 0, "mut_reject_plus_rate": 0,
             "plus_ok": 0, "by_source": {}}
    for pi, item in enumerate(pool):
        if len(tasks) >= n_tasks:
            break
        # 每题一行细粒度进度：静默死亡时最后一条即凶手（M1 构建三连死取证）
        print(f"[build] {pi}/{len(pool)} built={len(tasks)} tid={item['tid']} src={item['source']}", flush=True)
        ref_code = item["ref_code"]
        visible_test = item["visible_test"]
        harness_prefix, atol, fn_name = item["harness_prefix"], item["atol"], item["fn_name"]

        chosen = None
        for mut in make_mutants(ref_code, rng):
            ok, _ = _try_exec_visible(mut, visible_test)
            if ok is not False:        # True（可见测试过了）或 TIMEOUT → 弃
                stats["mut_reject_visible_pass"] += 1
                continue
            rate, n_plus = _plus_rate(mut, harness_prefix, fn_name, atol)
            if rate is None:
                rate, n_plus = 0.0, 0
            if not (min_plus_rate <= rate < max_plus_rate):
                stats["mut_reject_plus_rate"] += 1
                continue
            chosen = (mut, rate, n_plus)
            break
        if chosen is None:
            stats["no_valid_mutant"] += 1
            continue
        mut, mut_plus_rate, n_plus = chosen

        # 用真实沙盒生成"失败输出"展示文本（与 rollout 反馈同格式）
        r = sandbox_exec(mut + "\n\ntry:\n    " + visible_test +
                         "\nexcept Exception:\n    import traceback; traceback.print_exc(limit=2)\n",
                         timeout=5)
        fail_output = (r["stdout"] + r["stderr"]).strip()[-800:] or "(no output)"

        tid = item["tid"]
        tasks.append({
            "task_id": tid, "source": item["source"],
            "prompt": item["prompt"], "func_name": fn_name,
            "buggy_code": mut, "visible_test": visible_test,
            "fail_output": fail_output, "n_plus": n_plus,
            "mut_plus_rate": round(mut_plus_rate, 4),  # 难度标签，不进 prompt
        })
        hidden.append({
            "task_id": tid, "ref_code": ref_code, "fn_name": fn_name,
            "ref_norm": norm_ws(ref_code),   # rollout 查重用（与参考解相同判 0）
            "orig_tests": item["orig_tests"],
            "harness_prefix": harness_prefix if item["plus_ok"] else "", "atol": atol,
            "plus_ok": item["plus_ok"],
        })
        stats["built"] += 1
        stats["plus_ok"] += int(item["plus_ok"])
        stats["by_source"][item["source"]] = stats["by_source"].get(item["source"], 0) + 1

    with open(os.path.join(out_dir, "tasks.jsonl"), "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "hidden.jsonl"), "w") as f:
        for h in hidden:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(json.dumps({"stats": stats, "out": out_dir}, ensure_ascii=False))
    return tasks, hidden


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/c4")
    ap.add_argument("--parquets", default=f"{PARQUET},{PARQUET_HE}")
    args = ap.parse_args()
    build(args.n, args.seed, args.out, tuple(args.parquets.split(",")))
