"""C4代码执行环境 + 奖励函数 —— 与 m3_env 同构：把 <search> 换成 <run_code>

设计要点：
1. 沙盒：子进程 + unshare -rn 网络命名空间隔离（防远程泄题，Georgia Tech 2026
   实证的头号攻击面）+ RLIMIT_CPU/AS/FSIZE + 输出截断 + 无网络兜底降级。
2. 稠密奖励：EvalPlus(MBPP+) 的 plus harness 是数据驱动（inputs/results 数组），
   grading 时逐 case try/except → per-case 通过率（几十个 case，比 0/1 稠密得多）。
3. 终止门控（检索 Agent 项目 run1 教训的移植）：r_pass 只在合法 <answer> 终止时发放；
   未终止/非法终止 → 0。防"修而不交"轨迹拿分。
4. 防篡改：隐藏测试不进入沙盒文件系统，grading 独立子进程执行，
   模型代码物理上接触不到测试文件与参考解。

与 m3_env 的接口对应：
  SearchEnv.search(query)      ->  CodeExecEnv.run_visible(code)   # 轮内工具调用
  compute_reward(...)          ->  compute_reward_code(...)        # 轨迹奖励
  parse_trajectory(...)        ->  parse_trajectory_code(...)      # <run_code>/<answer>
"""
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile

# ---------------- 常量 ----------------
MAX_RUNS = 5       # run_code 次数硬上限（与 SYSTEM 提示、奖励惩罚一致）
MAX_TURNS = 6      # 最多 6 个 assistant 轮（5 run + 1 answer）
TIMEOUT_VISIBLE = 5      # 轮内执行超时（秒）
TIMEOUT_GRADE = 20       # 隐藏测试 grading 超时（秒，plus case 多）
MAX_OUT_CHARS = 1200     # 反馈给模型的 stdout/stderr 截断长度
MEM_LIMIT_BYTES = 4 * 1024 ** 3   # RLIMIT_AS 4GB（numpy 需要较宽 VA 空间）

SYSTEM = (
    "You are a code-repair agent. You are given a buggy Python function and one failing visible test.\n"
    "To execute code and see the visible test result, output exactly: <run_code>your code</run_code>\n"
    "Put the full function definition inside <run_code>; the environment appends the visible test.\n"
    "The sandbox has no internet access and cannot write files.\n"
    "When confident, submit the final fixed function, output exactly: <answer>```python\nyour full fixed function\n```</answer>\n"
    f"Run code at most {MAX_RUNS} times. The final grade uses hidden tests, not the visible one."
)

RUN_RE = re.compile(r"<run_code>(.*?)(?:</run_code>|$)", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S)
PYFENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


# ---------------- 沙盒执行 ----------------
_UNSHARE_OK = None  # 三态：None 未探测 / True 可用 / False 不可用


def _netns_available() -> bool:
    """探测 unshare -rn 是否可用（宿主机可用；docker 默认 seccomp 拦截，
    需 --security-opt seccomp=unconfined）。结果缓存。"""
    global _UNSHARE_OK
    if _UNSHARE_OK is None:
        if not shutil.which("unshare"):
            _UNSHARE_OK = False
        else:
            try:
                r = subprocess.run(["unshare", "-rn", "true"],
                                   capture_output=True, timeout=10)
                _UNSHARE_OK = (r.returncode == 0)
            except Exception:
                _UNSHARE_OK = False
    return _UNSHARE_OK


def _limits(timeout_s: int):
    def _set():
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s + 1))
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))   # 禁写任何文件
    return _set


def sandbox_exec(script_src: str, timeout: int, workdir: str = None):
    """在隔离子进程里执行一段 Python 源码。
    返回 dict(rc, stdout, stderr, timed_out, netns)。
    防御层：netns 禁网 / RLIMIT 限时限内存禁写文件 / 输出截断 / 最小环境变量。"""
    tmp = tempfile.mkdtemp(prefix="c4sbx_", dir=workdir)
    path = os.path.join(tmp, "prog.py")
    with open(path, "w") as f:
        f.write(script_src)
    cmd = [sys.executable, "-B", "-I", path]   # -I 隔离模式：忽略用户 site/环境变量
    netns = _netns_available()
    if netns:
        cmd = ["unshare", "-rn"] + cmd
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": tmp, "TMPDIR": tmp,
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    }
    timed_out = False
    try:
        r = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True,
                           text=True, timeout=timeout, preexec_fn=_limits(timeout))
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        timed_out, rc = True, -9
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        err = (err + "\n[TIMEOUT] execution exceeded %ss" % timeout).strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"rc": rc, "stdout": out[:MAX_OUT_CHARS], "stderr": err[:MAX_OUT_CHARS],
            "timed_out": timed_out, "netns": netns}


# ---------------- 轮内工具调用：可见测试执行 ----------------
class CodeExecEnv:
    """与 SearchEnv 对应：run_visible() 即本环境的"工具调用"。"""

    def __init__(self, lang="en"):
        self.lang = lang

    def run_visible(self, code: str, visible_test: str):
        """执行模型代码 + 追加可见测试断言，返回 (反馈文本, info)。
        反馈文本进入下一轮 user message（与 m3 的检索结果同位）。"""
        script = (
            code + "\n\n"
            "# --- visible test ---\n"
            "try:\n"
            f"    {visible_test}\n"
            "    print('VISIBLE_TEST: PASSED')\n"
            "except Exception as e:\n"
            "    import traceback\n"
            "    print('VISIBLE_TEST: FAILED')\n"
            "    traceback.print_exc(limit=2)\n"
        )
        r = sandbox_exec(script, TIMEOUT_VISIBLE)
        text = (r["stdout"] + ("\n" + r["stderr"] if r["stderr"].strip() else "")).strip()
        passed = "VISIBLE_TEST: PASSED" in r["stdout"]
        info = {"passed": passed, "timed_out": r["timed_out"], "rc": r["rc"]}
        if r["timed_out"]:
            text = (text + "\n[execution timed out]").strip()
        feedback = f"Execution result:\n{text or '(no output)'}"
        return feedback, info


# ---------------- 隐藏测试 grading ----------------
def _grade_script(solution_code: str, orig_tests: list, harness_src: str,
                  fn_name: str, atol: float) -> str:
    """构造 grading 脚本：先逐条跑原始 assert，再逐 case 跑 plus harness。
    结果以 GRADE_JSON 行输出**聚合计数**（恒定长度——初版输出 0/1 数组，
    HumanEval+ 上千 case 时超出 MAX_OUT_CHARS 截断致 JSON 解析失败，全判 0）。
    任何一段失败不影响另一段（各自 try/except）。"""
    return f'''
import json, sys
_ns = {{"__name__": "__main__"}}
# --- 1) 载入被测代码（可能本身抛异常/死循环；死循环由外层超时兜底）---
_code_ok = True
try:
    exec({json.dumps(solution_code)}, _ns)
except Exception:
    _code_ok = False
_o_pass = _o_n = _p_pass = _p_n = 0
# --- 2) 原始 assert 逐条 ---
for t in {json.dumps(orig_tests)}:
    _o_n += 1
    if not _code_ok:
        continue
    try:
        exec(t, dict(_ns))
        _o_pass += 1
    except Exception:
        pass
# --- 3) plus harness 逐 case ---
if _code_ok and {json.dumps(harness_src)}:
    try:
        h = {{"__name__": "__harness__"}}
        exec({json.dumps(harness_src)}, h)
        fn = h[{json.dumps(fn_name)}] if {json.dumps(fn_name)} in h else _ns[{json.dumps(fn_name)}]
        inputs, results, assertion = h["inputs"], h["results"], h["assertion"]
        for inp, exp in zip(inputs, results):
            _p_n += 1
            try:
                assertion(fn(*inp), exp, {json.dumps(atol)})
                _p_pass += 1
            except Exception:
                pass
    except Exception:
        _p_pass = _p_n = 0  # harness 解析失败 → plus 为空，退化为仅原始 assert
print("GRADE_JSON:" + json.dumps({{"code_ok": _code_ok, "orig_pass": _o_pass,
                                   "orig_n": _o_n, "plus_pass": _p_pass, "plus_n": _p_n}}))
'''


def grade_solution(solution_code: str, hidden: dict) -> dict:
    """对最终提交代码跑隐藏测试。hidden = {orig_tests, harness_prefix, fn_name, atol}
    返回 {orig_rate, plus_rate, n_plus, code_ok, timed_out, pass_rate}
    pass_rate（主奖励）：有 plus 用 plus，否则退化用 orig。"""
    script = _grade_script(solution_code, hidden["orig_tests"],
                           hidden.get("harness_prefix", ""), hidden["fn_name"],
                           hidden.get("atol", 0))
    r = sandbox_exec(script, TIMEOUT_GRADE)
    out = {"orig_rate": 0.0, "plus_rate": None, "n_plus": 0,
           "code_ok": False, "timed_out": r["timed_out"], "pass_rate": 0.0}
    m = re.search(r"GRADE_JSON:(\{.*\})", r["stdout"])
    if m:
        try:
            g = json.loads(m.group(1))
            out["code_ok"] = bool(g.get("code_ok"))
            if g.get("orig_n"):
                out["orig_rate"] = g.get("orig_pass", 0) / g["orig_n"]
            if g.get("plus_n"):
                out["plus_rate"] = g.get("plus_pass", 0) / g["plus_n"]
                out["n_plus"] = g["plus_n"]
        except Exception:
            pass
    out["pass_rate"] = out["plus_rate"] if out["plus_rate"] is not None else out["orig_rate"]
    return out


# ---------------- 解析与奖励 ----------------
def parse_trajectory_code(text_turns):
    """把每个 assistant 轮分类为 run / answer / invalid（与 m3.parse_trajectory 同构）"""
    parsed = []
    for raw in text_turns:
        m = ANSWER_RE.search(raw)
        if m:
            parsed.append(("answer", m.group(1).strip(), raw))
            continue
        m = RUN_RE.search(raw)
        if m:
            parsed.append(("run", m.group(1).strip(), raw))
            continue
        parsed.append(("invalid", None, raw))
    return parsed


def extract_code(answer_content: str):
    """从 <answer> 内容提取代码：优先 ```python 围栏，否则剥掉首尾非代码行"""
    m = PYFENCE_RE.search(answer_content)
    if m:
        return m.group(1).strip()
    # 无围栏：取 <answer> 原文（模型可能直接贴代码）
    txt = answer_content.strip()
    return txt if txt else None


def compute_reward_code(text_turns, grade_info,
                        w_pass=1.0, w_format=0.2, w_over=0.2):
    """R = w_pass*隐藏测试通过率 + w_format*格式 − w_over*超限
    终止门控：grade_info 为 None（未产生合法 <answer>）时 r_pass=0。
    （检索 Agent 项目 run1 教训：过程性信号必须由合法终止门控，防"修而不交"白拿分）"""
    parsed = parse_trajectory_code(text_turns)
    all_valid = all(k in ("run", "answer") for k, _, _ in parsed)
    ends_answer = bool(parsed) and parsed[-1][0] == "answer"
    only_one_answer = sum(1 for k, _, _ in parsed if k == "answer") <= 1
    code_ok = grade_info is not None and grade_info.get("code_extracted")
    r_format = float(all_valid and ends_answer and only_one_answer and code_ok)

    r_pass = grade_info["pass_rate"] if (grade_info and code_ok) else 0.0

    n_runs = sum(1 for k, _, _ in parsed if k == "run")
    over = max(0, n_runs - MAX_RUNS) + max(0, len(parsed) - MAX_TURNS)
    r_over = float(min(over, 2)) / 2.0

    total = w_pass * r_pass + w_format * r_format - w_over * r_over
    info = {
        "r_pass": round(r_pass, 4), "r_format": r_format, "r_over": r_over,
        "n_runs": n_runs, "total": round(total, 4),
        "plus_rate": (grade_info or {}).get("plus_rate"),
        "orig_rate": (grade_info or {}).get("orig_rate"),
        "timed_out": (grade_info or {}).get("timed_out", False),
    }
    return total, info
