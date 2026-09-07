"""PoC-A（WS-1）：vLLM 深度定制实证 benchmark —— 多轮锁步 rollout 负载

工作负载 = 复用 c4_rollout.rollout() 的真实评测负载（默认 89 题 × G1 × temp0 贪心，
模型 = checkpoints/c4_grpo/iter1_merged，即头条 0.7516 评测的确切负载），
通过薄代理 wrap LLM.generate() 做逐调用计时 / token 计数 / 前缀命中解析——
c4_rollout.py 零改动（rollout(llm=...) 注入口已有）。

预注册判据（2026-09-06，见迭代计划）：
  成功 = 任一配置在不违反不变性判据（|Δavg_pass|≤0.01 且翻转题≤2/89、翻转可归因
         fp 归约平局）下总 wall-clock 降 ≥20%；
         或产出"sm_70 上 X 不可用 + traceback + 降级路径"完整实证（工程叙事同为成功）。
  失败 = 全配置差 <5% 且默认已开满 → 如实写"该负载默认配置已近优，
         定制空间在换载/并行层"。
  保真锚点 = config0（与历史评测同配置）应复现 avg_pass≈0.7516（temp0 跨进程
         不保证 bit 级一致，容差 ±0.01 预注册在案）。

口径纪律：
  - 聚合吞吐与单流吞吐分开记录（单流 23/26 tok/s 的旧数字不能外推批量负载）；
  - prefix cache 命中率三路互证：引擎周期日志（stdout grep）/ analytic LCP 法 /
    引擎 metrics（若可用），对不上以 analytic 为准并记录差异；
  - 显存峰值用 nvidia-smi 后台采样（torch.cuda.max_memory_allocated 对 vLLM
    自有分配器无效）；
  - 头条数字（a1）串行单卡跑，防 CPU 沙盒子进程争抢污染 wall-clock。

用法（docker 内，见文件尾 example）：
  # A0 冒烟（探针彼此独立进程，崩溃不连坐）
  python poc_vllm_bench.py --stage a0 --probe cudagraph --n 8
  python poc_vllm_bench.py --stage a0 --probe prefix_off --n 8
  python poc_vllm_bench.py --stage a0 --probe prefix_on  --n 8
  python poc_vllm_bench.py --stage a0 --probe util       --n 8
  # A1 头条（每配置一个独立进程，外层 bash for 循环）
  python poc_vllm_bench.py --stage a1 --config 0
  # 对比与不变性核对（纯 json，可在宿主机跑）
  python3 poc_vllm_bench.py --stage compare
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback

OUT_DIR = "logs/poc_vllm"
DEFAULT_MODEL = "checkpoints/c4_grpo/iter1_merged"

# A1 配置矩阵（预注册）：
#   prefix: None=不动引擎默认（V1 默认开启，须以有效配置打印件为准）/ True / False
#   eager5: None 表示跟随 A0 胜出者（跑 a1 前用 --eager5 显式传入）
CONFIGS = {
    0: dict(label="现状基线(默认prefix+eager+util0.7)", prefix=None, eager=True,  util=0.7),
    1: dict(label="prefix显式ON",                       prefix=True, eager=True,  util=0.7),
    2: dict(label="prefix显式OFF",                      prefix=False, eager=True, util=0.7),
    3: dict(label="CUDAgraph(eager=OFF)",               prefix=None, eager=False, util=0.7),
    4: dict(label="prefixON+CUDAgraph",                 prefix=True, eager=False, util=0.7),
    5: dict(label="prefixON+util0.9(+eager随A0)",       prefix=True, eager=None,  util=0.9),
}


# ---------------------------------------------------------------- utilities
def _norm_ws(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


class MemSampler:
    """nvidia-smi 后台采样显存峰值（2s 间隔）；vLLM 自有分配器下唯一可靠口径。"""

    def __init__(self, interval=2.0):
        self.interval = interval
        self.peak = 0
        self.samples = []
        self._stop = threading.Event()
        self._th = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits", "-i", "0"],
                    timeout=5).decode().strip().splitlines()
                used = int(out[0])
                self.peak = max(self.peak, used)
                self.samples.append((round(time.time(), 1), used))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._th.join(timeout=5)


def dump_engine_config(llm):
    """打印/返回引擎**有效配置**（prefix caching 默认态必须以实际值为准）。"""
    info = {}
    try:
        eng = getattr(llm, "llm_engine", None)
        vc = getattr(eng, "vllm_config", None)
        if vc is not None:
            cc = getattr(vc, "cache_config", None)
            mc = getattr(vc, "model_config", None)
            sc = getattr(vc, "scheduler_config", None)
            pc = getattr(vc, "parallel_config", None)
            info = {
                "enable_prefix_caching": getattr(cc, "enable_prefix_caching", "?"),
                "block_size": getattr(cc, "block_size", "?"),
                "gpu_memory_utilization": getattr(cc, "gpu_memory_utilization",
                                                  getattr(vc, "gpu_memory_utilization", "?")),
                "enforce_eager": getattr(mc, "enforce_eager", "?"),
                "max_num_seqs": getattr(sc, "max_num_seqs", "?"),
                "max_model_len": getattr(mc, "max_model_len", "?"),
                "dtype": str(getattr(mc, "dtype", "?")),
                "tensor_parallel_size": getattr(pc, "tensor_parallel_size", "?"),
                "cudagraph_mode": str(getattr(vc, "compilation_config", ""))[:200],
            }
    except Exception as e:
        info["_error"] = repr(e)
    print("EFFECTIVE_ENGINE_CONFIG " + json.dumps(info, ensure_ascii=False), flush=True)
    return info


def build_llm(model, prefix, eager, util, max_model_len=8192):
    """按配置构造 vLLM；prefix=None 时不传该参数（记录引擎默认态）。"""
    from vllm import LLM
    kw = dict(dtype="float16", gpu_memory_utilization=util,
              max_model_len=max_model_len, enforce_eager=eager)
    if prefix is not None:
        kw["enable_prefix_caching"] = prefix
    t0 = time.perf_counter()
    llm = LLM(model=model, **kw)
    load_wall = time.perf_counter() - t0
    return llm, load_wall, kw


class LLMBenchProxy:
    """薄代理：wrap .generate() 逐调用计时/token 计数/analytic 前缀命中。

    analytic 命中口径：对每条新 prompt，在历史 (prompt_ids+output_ids) 上下文集合中
    求最长公共 token 前缀（LCP）的最大值 = 理论可命中上限；引擎实际命中以周期日志
    grep 为准，两者对照记录。锁步多轮下第 k>1 轮的 prompt 应为上一轮上下文的严格
    前缀扩展（若 chat template 重渲染破坏了前缀性，LCP 会如实暴露——这本身是
    prefix caching 在多轮 agent 负载上的关键实测点）。
    """

    def __init__(self, llm):
        self._llm = llm
        self.calls = []
        self._contexts = []   # 历史 prompt_ids+output_ids
        self._t_gen_total = 0.0
        self.overhead_total = 0.0   # 代理自身（计时/LCP 解析）开销，从 rollout_wall 中扣除

    def get_tokenizer(self):
        return self._llm.get_tokenizer()

    @staticmethod
    def _lcp(a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def generate(self, prompts, sp):
        t0 = time.perf_counter()
        outs = self._llm.generate(prompts, sp)
        wall = time.perf_counter() - t0
        self._t_gen_total += wall
        t_ov = time.perf_counter()
        p_ids = [list(o.prompt_token_ids) for o in outs]
        c_ids = [list(o.outputs[0].token_ids) for o in outs]
        # analytic LCP（对每条新 prompt 在历史上下文里找最优前缀）
        hits = []
        for p in p_ids:
            best = 0
            head = p[:32]
            for ctx in self._contexts:
                if len(ctx) < len(head) or ctx[:32] != head:
                    continue
                l = self._lcp(p, ctx)
                if l > best:
                    best = l
            hits.append(best)
        rec = {
            "call": len(self.calls), "wall": round(wall, 3), "n_prompts": len(prompts),
            "prompt_tokens": sum(len(p) for p in p_ids),
            "completion_tokens": sum(len(c) for c in c_ids),
            "analytic_hit_tokens": sum(hits),
            "per_prompt_len": [len(p) for p in p_ids],
            "per_prompt_hit": hits,
            "per_prompt_new": [len(c) for c in c_ids],
        }
        self.calls.append(rec)
        self._contexts += [p + c for p, c in zip(p_ids, c_ids)]
        self.overhead_total += time.perf_counter() - t_ov
        print(f"  [gen call {rec['call']}] wall={rec['wall']}s n={rec['n_prompts']} "
              f"prompt_tok={rec['prompt_tokens']} compl_tok={rec['completion_tokens']} "
              f"analytic_hit={rec['analytic_hit_tokens']}", flush=True)
        return outs

    def __getattr__(self, name):   # 其余属性透传
        return getattr(self._llm, name)


def summarize(rec, results, cfg_label, extra=None):
    n = len(results)
    avg_pass = sum((r["grade"] or {}).get("pass_rate") or 0 for r in results) / n
    gen_wall = sum(c["wall"] for c in rec.calls)
    ptok = sum(c["prompt_tokens"] for c in rec.calls)
    ctok = sum(c["completion_tokens"] for c in rec.calls)
    hit = sum(c["analytic_hit_tokens"] for c in rec.calls)
    s = {
        "config_label": cfg_label,
        "n_traj": n,
        "avg_pass": round(avg_pass, 4),
        "submit_rate": round(sum(1 for r in results if r["answer_code"]) / n, 4),
        "avg_runs": round(sum(r["n_runs"] for r in results) / n, 3),
        "rollout_wall_total": round(rec.rollout_wall, 2),
        "generate_wall_total": round(gen_wall, 2),
        "proxy_overhead_wall": round(rec.proxy_overhead, 2),
        "sandbox_wall_clean": round(rec.rollout_wall - gen_wall - rec.proxy_overhead, 2),
        "engine_load_wall": round(rec.engine_load_wall, 2),
        "prompt_tokens_total": ptok,
        "completion_tokens_total": ctok,
        "agg_output_tok_per_s": round(ctok / gen_wall, 2) if gen_wall else None,
        "agg_prompt_tok_per_s": round(ptok / gen_wall, 2) if gen_wall else None,
        "analytic_cached_frac": round(hit / ptok, 4) if ptok else None,
        "mem_peak_mib": rec.mem_peak,
        "per_call": rec.calls,
        **(extra or {}),
    }
    return s, avg_pass


def save_results(path, results):
    """不变性核对用：逐题 pass_rate / answer_code(norm_ws) / turn_texts。"""
    slim = [{
        "task_id": r["task_id"],
        "pass_rate": (r["grade"] or {}).get("pass_rate"),
        "dup_ref": bool((r["grade"] or {}).get("dup_ref")),
        "answer_norm": _norm_ws(r.get("answer_code")),
        "n_runs": r["n_runs"],
        "turn_texts": r["turn_texts"],
    } for r in results]
    with open(path, "w") as f:
        json.dump(slim, f, ensure_ascii=False)


class _Rec:  # 汇总容器
    pass


def run_workload(model, n_tasks, cfg_label, prefix, eager, util, out_prefix,
                 tasks_path="data/c4/tasks_train.jsonl",
                 hidden_path="data/c4/hidden_train.jsonl",
                 G=1, temperature=0.0):
    """构造 LLM → 代理 → 真实 rollout 负载 → 汇总落盘。返回 (summary, ok)。"""
    from c4_rollout import rollout, load_tasks
    tasks, hidden = load_tasks(tasks_path, hidden_path, n_tasks)
    rec = _Rec()
    mem = MemSampler()
    try:
        with mem:
            llm, load_wall, kw = build_llm(model, prefix, eager, util)
            rec.engine_load_wall = load_wall
            eff = dump_engine_config(llm)
            proxy = LLMBenchProxy(llm)
            t0 = time.perf_counter()
            results, _ = rollout(model, tasks, hidden, G=G, temperature=temperature,
                                 llm=proxy)
            rec.rollout_wall = time.perf_counter() - t0
            rec.proxy_overhead = proxy.overhead_total
            rec.calls = proxy.calls
        rec.mem_peak = mem.peak
        s, avg_pass = summarize(rec, results, cfg_label,
                                extra={"build_kwargs": {k: str(v) for k, v in kw.items()},
                                       "effective_engine_config": eff})
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(f"{OUT_DIR}/{out_prefix}.json", "w") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
        save_results(f"{OUT_DIR}/{out_prefix}_results.json", results)
        print("SUMMARY " + json.dumps({k: v for k, v in s.items()
                                       if k not in ("per_call",)}, ensure_ascii=False),
              flush=True)
        print(f"avg_pass={avg_pass:.4f} (保真锚点 0.7516, 容差 ±0.01)", flush=True)
        return s, True
    except Exception:
        tb = traceback.format_exc()
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(f"{OUT_DIR}/{out_prefix}_FAIL.json", "w") as f:
            json.dump({"config_label": cfg_label, "traceback": tb,
                       "mem_peak_mib": mem.peak}, f, ensure_ascii=False, indent=1)
        print("PROBE_FAIL " + tb[-2000:], flush=True)
        return {"config_label": cfg_label, "fail": True, "traceback": tb[-2000:]}, False


# ---------------------------------------------------------------- stages
def stage_a0(args):
    model = args.model
    if args.probe == "cudagraph":
        run_workload(model, args.n, "A0 cudagraph(eager=False, 默认prefix, util0.7)",
                     None, False, 0.7, "a0_cudagraph")
    elif args.probe == "prefix_off":
        run_workload(model, args.n, "A0 prefix显式OFF可用性", False, True, 0.7,
                     "a0_prefix_off")
    elif args.probe == "prefix_on":
        run_workload(model, args.n, "A0 prefix显式ON", True, True, 0.7, "a0_prefix_on")
    elif args.probe == "util":
        for u in (0.75, 0.8, 0.85, 0.9):
            ok = run_workload(model, args.n, f"A0 util阶梯 {u}", None, True, u,
                              f"a0_util_{str(u).replace('.', '')}")[1]
            if not ok:
                print(f"util 上限 = 上一档（{u} 失败）", flush=True)
                break
    else:
        raise SystemExit(f"unknown probe {args.probe}")


def stage_a1(args):
    cfg = dict(CONFIGS[args.config])
    eager = cfg["eager"]
    if eager is None:
        eager = bool(args.eager5)
        print(f"config5 eager 随 A0 胜出者: {eager}", flush=True)
    run_workload(args.model, args.n, f"A1 config{args.config} {cfg['label']}",
                 cfg["prefix"], eager, cfg["util"], f"a1_config{args.config}")


def stage_compare(args):
    """对比矩阵 + 不变性核对（纯 json，宿主机可跑）。基线 = a1_config0。"""
    base_path = f"{OUT_DIR}/a1_config0_results.json"
    if not os.path.exists(base_path):
        raise SystemExit("config0 结果不存在，先跑 a1")
    base = json.load(open(base_path))
    rows, invariance = [], []
    for c in sorted(CONFIGS):
        p = f"{OUT_DIR}/a1_config{c}.json"
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        rows.append({k: s.get(k) for k in
                     ("config_label", "avg_pass", "generate_wall_total",
                      "rollout_wall_total", "proxy_overhead_wall", "sandbox_wall_clean",
                      "engine_load_wall", "agg_output_tok_per_s",
                      "agg_prompt_tok_per_s", "analytic_cached_frac", "mem_peak_mib",
                      "prompt_tokens_total", "completion_tokens_total")})
        rp = f"{OUT_DIR}/a1_config{c}_results.json"
        if c != 0 and os.path.exists(rp):
            cur = json.load(open(rp))
            flips, dpass, code_mismatch = [], 0.0, 0
            for b, x in zip(base, cur):
                assert b["task_id"] == x["task_id"]
                bp = b["pass_rate"] or 0.0
                xp = x["pass_rate"] or 0.0
                dpass += abs(xp - bp)
                if abs(xp - bp) > 1e-9:
                    flips.append((b["task_id"], round(bp, 3), round(xp, 3)))
                if b["answer_norm"] != x["answer_norm"]:
                    code_mismatch += 1
            invariance.append({
                "config": c, "avg_abs_pass_diff": round(dpass / len(base), 5),
                "n_flipped_tasks": len(flips), "flips": flips[:10],
                "answer_code_mismatch": code_mismatch,
                "verdict_pre_registered": ("PASS" if abs(dpass / len(base)) <= 0.01
                                           and len(flips) <= 2 else "REVIEW"),
            })
    out = {"matrix": rows, "invariance_vs_config0": invariance}
    with open(f"{OUT_DIR}/compare.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    # 头条口径 = generate_wall_total（纯引擎时间，剔除代理开销与沙盒争抢，跨配置最干净）
    if rows:
        w0 = rows[0].get("generate_wall_total")
        for r in rows[1:]:
            gw = r.get("generate_wall_total")
            if gw and w0:
                d = (gw - w0) / w0 * 100
                print(f"generate_wall vs 基线: {r['config_label']}: {d:+.1f}% "
                      f"(预注册成功线 ≤ -20%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["a0", "a1", "compare"])
    ap.add_argument("--probe", default="cudagraph",
                    choices=["cudagraph", "prefix_off", "prefix_on", "util"])
    ap.add_argument("--config", type=int, default=0)
    ap.add_argument("--eager5", type=int, default=1, help="config5 的 eager（随 A0 胜出者）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=89)
    args = ap.parse_args()
    if args.stage == "a0":
        stage_a0(args)
    elif args.stage == "a1":
        stage_a1(args)
    else:
        stage_compare(args)

# A0（docker detached，探针独立进程互不连坐）：
#   docker run -d --name pocA0 --gpus '"device=0"' --security-opt seccomp=unconfined \
#     -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
#     -e HF_HUB_OFFLINE=1 1cat-vllm:v100-1.3.0 \
#     bash -c 'mkdir -p logs/poc_vllm; for p in cudagraph prefix_off prefix_on util; do
#       echo "=== PROBE $p ==="; python -u poc_vllm_bench.py --stage a0 --probe $p --n 8;
#       echo "probe $p exit=$?"; done > logs/poc_vllm/a0_all.log 2>&1'
# A1（同样 detached 串行，每配置独立进程）：
#   bash -c 'for c in 0 1 2 3 4 5; do echo "=== CONFIG $c ===";
#     python -u poc_vllm_bench.py --stage a1 --config $c --eager5 <A0胜出>;
#     echo "config $c exit=$?"; done > logs/poc_vllm/a1_all.log 2>&1'
