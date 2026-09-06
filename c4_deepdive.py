"""C4 深挖分析：'学会修'行为指标 + checkpoint 题级 diff（M3 行为曲线的细化）

Phase 1（纯日志，无依赖）：
  从 logs/c4_*multiturn*.json 提取每条轨迹的 run 反馈序列（VISIBLE_TEST: FAILED/PASSED），
  计算方案 §三"良好"标准点名的"二次提交通过率"族指标：
    - 修复成功率：出现过 FAILED 反馈的轨迹中，终版隐藏测试满分(pass_rate==1.0)的比例
    - 首跑失败修复率：首跑即 FAILED 的轨迹中，终版满分的比例（"二次提交通过率"字面口径）
    - 可见修复率：FAILED 后曾把可见测试修到 PASSED 的比例（反馈被真实利用的证据）
  以及两个 checkpoint 的题级 diff（峰值 iter1 vs 平台 iter3：过冲发生在哪些题）。

Phase 2（--phase2，需沙盒环境，docker 内跑）：
  对 n_runs>=1 的轨迹，把**首跑 run_code 的代码**用隐藏测试离线判分（c4_env.grade_solution），
  得 first_rate；与终版 pass_rate 对比 = 多轮行为的真实修复增量（隐藏口径，含部分分）。

用法：
  python3 c4_deepdive.py                 # Phase 1
  python3 c4_deepdive.py --phase2        # Phase 1+2（在带沙盒的容器内）
"""
import argparse
import json
import os
import re
import sys

LOGS = [
    ("基座多轮",   "logs/c4_baseline_multiturn.json"),
    ("RFT多轮",    "logs/c4_rft_multiturn_train89.json"),
    ("GRPO-iter0", "logs/c4_grpo_iter0_multiturn.json"),
    ("GRPO-iter1", "logs/c4_grpo_iter1_multiturn.json"),
    ("GRPO-iter2", "logs/c4_grpo_iter2_multiturn.json"),
    ("GRPO-iter3", "logs/c4_grpo_iter3_multiturn.json"),
]

VIS_RE = re.compile(r"VISIBLE_TEST:\s*(PASSED|FAILED)")


def run_feedback_seq(traj):
    """按轮提取 run 反馈的可见测试结果序列（'P'/'F'），只看环境反馈 user 消息。"""
    seq = []
    for m in traj["messages"]:
        if m["role"] != "user" or not m["content"].startswith("Execution result:"):
            continue
        found = VIS_RE.findall(m["content"])
        # 一条反馈可能含多个可见断言输出时取首个总判定；无判定（如语法错误截断）记 'E'
        seq.append(found[0][0] if found else "E")
    return seq


def phase1():
    out = {}
    for name, path in LOGS:
        if not os.path.exists(path):
            print(f"[skip] {path} 不存在")
            continue
        rs = json.load(open(path))
        n = len(rs)
        ran = [r for r in rs if r["n_runs"] >= 1]
        saw_fail, first_fail, fixed_vis = [], [], 0
        for r in ran:
            seq = run_feedback_seq(r)
            bad = next((i for i, x in enumerate(seq) if x in ("F", "E")), None)
            if bad is not None:
                saw_fail.append(r)
                # 失败反馈之后出现过 PASSED = 反馈被真实利用、可见测试被修好
                if "P" in seq[bad + 1:]:
                    fixed_vis += 1
            if seq and seq[0] in ("F", "E"):
                first_fail.append(r)

        def full_pass(subset):
            ok = [r for r in subset if (r["grade"] or {}).get("pass_rate") == 1.0]
            avg = (sum((r["grade"] or {}).get("pass_rate", 0.0) for r in subset) / len(subset)) if subset else 0.0
            return len(ok), avg

        f_ok, f_avg = full_pass(saw_fail)
        ff_ok, ff_avg = full_pass(first_fail)
        stat = {
            "n_traj": n,
            "n_ran": len(ran),
            "n_saw_fail": len(saw_fail),
            "repair_success_full": round(f_ok / len(saw_fail), 4) if saw_fail else None,
            "repair_success_avgpass": round(f_avg, 4) if saw_fail else None,
            "n_first_fail": len(first_fail),
            "second_submit_full": round(ff_ok / len(first_fail), 4) if first_fail else None,
            "second_submit_avgpass": round(ff_avg, 4) if first_fail else None,
            "visible_fixed_rate": round(fixed_vis / len(saw_fail), 4) if saw_fail else None,
        }
        out[name] = stat
        print(f"[{name}] {json.dumps(stat, ensure_ascii=False)}")
    return out


def task_diff(name_a, path_a, name_b, path_b):
    """题级 diff：pass_rate 差异与翻转题清单（带 n_runs/dup/timeout 上下文）。"""
    A = {r["task_id"]: r for r in json.load(open(path_a))}
    B = {r["task_id"]: r for r in json.load(open(path_b))}
    assert set(A) == set(B), "题集不一致"
    flips_ab, flips_ba, deltas = [], [], []
    for tid in A:
        pa = (A[tid]["grade"] or {}).get("pass_rate", 0.0)
        pb = (B[tid]["grade"] or {}).get("pass_rate", 0.0)
        deltas.append(pb - pa)
        if pa == 1.0 and pb < 1.0:
            flips_ab.append((tid, round(pa, 3), round(pb, 3), B[tid]["n_runs"],
                             (B[tid]["grade"] or {}).get("dup_ref", False),
                             (B[tid]["grade"] or {}).get("timed_out", False)))
        elif pb == 1.0 and pa < 1.0:
            flips_ba.append((tid, round(pa, 3), round(pb, 3)))
    print(f"\n=== 题级 diff {name_a} -> {name_b} ===")
    print(f"平均 pass_rate 变化: {sum(deltas)/len(deltas):+.4f}")
    print(f"{name_a}满分->{name_b}掉分: {len(flips_ab)} 题")
    for x in flips_ab:
        print("  ", x)
    print(f"{name_b}新满分(从{name_a}非满分): {len(flips_ba)} 题")
    for x in flips_ba:
        print("  ", x)
    return {"flips_down": flips_ab, "flips_up": flips_ba,
            "avg_delta": round(sum(deltas) / len(deltas), 4)}


def _norm_ws(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def phase2(rows_out="logs/c4_deepdive_p2_rows.jsonl"):
    """首跑代码隐藏判分：多轮行为的真实修复增量（隐藏口径）。需要沙盒（容器内跑）。

    方法学修正（v2）：
    - **对称 dup 门控**：终版 pass_rate 已被查重门判 0，首跑分若不同样门控，
      dup 轨迹会虚增负 delta（iter1 6 条 dup ≈ 贡献 −0.054/−0.056，几乎全部）。
      首跑代码与 ref_norm 归一化相同 → first_rate 同样置 0。
    - **按首跑可见结果分支**：PASSED 分支 = 验证通道（delta≈0 是应然）；
      FAILED 分支 = 真实修复价值所在，单独统计。
    - 逐行落盘，避免重复判分。
    """
    from c4_env import grade_solution
    hiddens = {}
    for l in open("data/c4/hidden_train.jsonl"):
        h = json.loads(l)
        hiddens[h["task_id"]] = h
    run_re = re.compile(r"<run_code>(.*?)(?:</run_code>|$)", re.S)
    # 只跑四个关键日志（基座/RFT/峰值/平台），控制判分时长
    keep = {"基座多轮", "RFT多轮", "GRPO-iter1", "GRPO-iter3"}
    summary = {}
    rows_all = []
    for name, path in LOGS:
        if name not in keep:
            continue
        if not os.path.exists(path):
            continue
        rs = json.load(open(path))
        rows = []
        for r in rs:
            if r["n_runs"] < 1:
                continue
            # 首跑代码 = 第一条 assistant 消息里的 <run_code> 块
            first_code = None
            for m in r["messages"]:
                if m["role"] == "assistant":
                    mm = run_re.search(m["content"])
                    if mm:
                        first_code = mm.group(1)
                        break
            if not first_code:
                continue
            h = hiddens[r["task_id"]]
            g = grade_solution(first_code, h)
            first_raw = g.get("pass_rate", 0.0)
            first_dup = _norm_ws(first_code) == h["ref_norm"]
            first_gated = 0.0 if first_dup else first_raw
            final = (r["grade"] or {}).get("pass_rate", 0.0)   # 已含 dup 门
            final_dup = bool((r["grade"] or {}).get("dup_ref"))
            # 首跑可见测试结果（分支依据）
            seq = run_feedback_seq(r)
            branch = seq[0] if seq else "?"   # P / F / E
            row = {"model": name, "task_id": r["task_id"], "branch": branch,
                   "first_raw": round(first_raw, 4), "first_dup": first_dup,
                   "first_gated": round(first_gated, 4), "final": round(final, 4),
                   "final_dup": final_dup, "n_runs": r["n_runs"]}
            rows.append(row)
            rows_all.append(row)
        if not rows:
            continue

        def agg(subset):
            if not subset:
                return None
            f0 = sum(x["first_gated"] for x in subset) / len(subset)
            f1 = sum(x["final"] for x in subset) / len(subset)
            return {"n": len(subset), "first_gated_avg": round(f0, 4),
                    "final_avg": round(f1, 4), "delta": round(f1 - f0, 4)}

        summary[name] = {
            "all": agg(rows),
            "branch_P_first_passed": agg([x for x in rows if x["branch"] == "P"]),
            "branch_F_first_failed": agg([x for x in rows if x["branch"] in ("F", "E")]),
            "first_dup_cnt": sum(1 for x in rows if x["first_dup"]),
            "final_dup_cnt": sum(1 for x in rows if x["final_dup"]),
        }
        print(f"[phase2 {name}] {json.dumps(summary[name], ensure_ascii=False)}", flush=True)
    with open(rows_out, "w") as f:
        for x in rows_all:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[phase2] per-task rows -> {rows_out} ({len(rows_all)})", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2", action="store_true")
    ap.add_argument("--out", default="logs/c4_deepdive.json")
    args = ap.parse_args()

    res = {"phase1": phase1()}
    res["diff_iter1_iter3"] = task_diff(
        "iter1", "logs/c4_grpo_iter1_multiturn.json",
        "iter3", "logs/c4_grpo_iter3_multiturn.json")
    if args.phase2:
        res["phase2"] = phase2()
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {args.out}")
