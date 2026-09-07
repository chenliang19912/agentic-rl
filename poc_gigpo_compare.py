"""WS-2 Stage 1.5 方向门对比器：配对 replay R0(epi) vs R1(gigpo)

口径完全复用 c4_deepdive.py（Phase1 修复指标族 / 题级 diff / Phase2 首跑隐藏判分
含对称 dup 门控与首跑可见分支），只把 LOGS 换成配对双臂文件。

预注册判据（c4_train_gigpo.py docstring，训练开跑前锁定）：
  主判据：R1−R0 贪心 avg_pass ≥ +0.005
  备选：  ≥ −0.005 且修复类指标（repair_success / visible_fixed_rate / F 分支 Δ）
          任一严格改善
  方向门：R1−R0 ≥ 0 才进 Stage 2，否则止损。

用法：
  python3 poc_gigpo_compare.py            # Phase1 + 题级 diff（宿主机可跑，纯 stdlib）
  python3 poc_gigpo_compare.py --phase2   # 加 Phase2 首跑隐藏判分（容器内，需 c4_env 沙盒）
产物：logs/poc_gigpo/stage15_compare.json
"""
import argparse
import json
import os
import re

from c4_deepdive import run_feedback_seq, _norm_ws

PAIR = [("R0_epi", "logs/poc_gigpo/eval_R0_multiturn.json"),
        ("R1_gigpo", "logs/poc_gigpo/eval_R1_multiturn.json")]
OUT = "logs/poc_gigpo/stage15_compare.json"


def phase1_pair():
    out = {}
    for name, path in PAIR:
        rs = json.load(open(path))
        n = len(rs)
        ran = [r for r in rs if r["n_runs"] >= 1]
        saw_fail, first_fail, fixed_vis = [], [], 0
        for r in ran:
            seq = run_feedback_seq(r)
            bad = next((i for i, x in enumerate(seq) if x in ("F", "E")), None)
            if bad is not None:
                saw_fail.append(r)
                if "P" in seq[bad + 1:]:
                    fixed_vis += 1
            if seq and seq[0] in ("F", "E"):
                first_fail.append(r)

        def full_pass(subset):
            ok = [r for r in subset if (r["grade"] or {}).get("pass_rate") == 1.0]
            avg = (sum((r["grade"] or {}).get("pass_rate", 0.0) for r in subset)
                   / len(subset)) if subset else 0.0
            return len(ok), avg

        f_ok, f_avg = full_pass(saw_fail)
        ff_ok, ff_avg = full_pass(first_fail)
        stat = {
            "n_traj": n,
            "avg_pass": round(sum((r["grade"] or {}).get("pass_rate", 0.0)
                                  for r in rs) / n, 4),
            "n_ran": len(ran),
            "n_saw_fail": len(saw_fail),
            "repair_success_full": round(f_ok / len(saw_fail), 4) if saw_fail else None,
            "repair_success_avgpass": round(f_avg, 4) if saw_fail else None,
            "n_first_fail": len(first_fail),
            "second_submit_full": round(ff_ok / len(first_fail), 4) if first_fail else None,
            "second_submit_avgpass": round(ff_avg, 4) if first_fail else None,
            "visible_fixed_rate": round(fixed_vis / len(saw_fail), 4) if saw_fail else None,
            "avg_runs": round(sum(r["n_runs"] for r in rs) / n, 3),
            "dup_cnt": sum(1 for r in rs if (r["grade"] or {}).get("dup_ref")),
        }
        out[name] = stat
        print(f"[{name}] {json.dumps(stat, ensure_ascii=False)}", flush=True)
    return out


def task_diff_pair():
    A = {r["task_id"]: r for r in json.load(open(PAIR[0][1]))}
    B = {r["task_id"]: r for r in json.load(open(PAIR[1][1]))}
    assert set(A) == set(B), "题集不一致"
    flips_down, flips_up, deltas = [], [], []
    for tid in sorted(A):
        pa = (A[tid]["grade"] or {}).get("pass_rate", 0.0)
        pb = (B[tid]["grade"] or {}).get("pass_rate", 0.0)
        deltas.append(pb - pa)
        if pa == 1.0 and pb < 1.0:
            flips_down.append((tid, round(pa, 3), round(pb, 3), B[tid]["n_runs"]))
        elif pb == 1.0 and pa < 1.0:
            flips_up.append((tid, round(pa, 3), round(pb, 3), A[tid]["n_runs"]))
    res = {"avg_delta": round(sum(deltas) / len(deltas), 4),
           "flips_down": flips_down, "flips_up": flips_up}
    print(f"\n=== 题级 diff R0 -> R1 ===")
    print(f"平均 pass_rate 变化: {res['avg_delta']:+.4f}")
    print(f"R0满分->R1掉分: {len(flips_down)} 题 {flips_down}")
    print(f"R1新满分: {len(flips_up)} 题 {flips_up}")
    return res


def phase2_pair():
    """首跑代码隐藏判分（对称 dup 门控 + 首跑可见分支），c4_deepdive.phase2 同口径。"""
    from c4_env import grade_solution
    hiddens = {}
    for l in open("data/c4/hidden_train.jsonl"):
        h = json.loads(l)
        hiddens[h["task_id"]] = h
    run_re = re.compile(r"<run_code>(.*?)(?:</run_code>|$)", re.S)
    summary = {}
    rows_all = []
    for name, path in PAIR:
        rs = json.load(open(path))
        rows = []
        for r in rs:
            if r["n_runs"] < 1:
                continue
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
            final = (r["grade"] or {}).get("pass_rate", 0.0)
            seq = run_feedback_seq(r)
            branch = seq[0] if seq else "?"
            row = {"model": name, "task_id": r["task_id"], "branch": branch,
                   "first_raw": round(first_raw, 4), "first_dup": first_dup,
                   "first_gated": round(first_gated, 4), "final": round(final, 4),
                   "final_dup": bool((r["grade"] or {}).get("dup_ref")),
                   "n_runs": r["n_runs"]}
            rows.append(row)
            rows_all.append(row)

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
    with open("logs/poc_gigpo/stage15_p2_rows.jsonl", "w") as f:
        for x in rows_all:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return summary


def verdict(p1, diff, p2):
    """预注册判据机械判定（不掺主观）。"""
    d = p1["R1_gigpo"]["avg_pass"] - p1["R0_epi"]["avg_pass"]
    repair_improved = []
    for k in ("repair_success_full", "repair_success_avgpass", "visible_fixed_rate",
              "second_submit_full", "second_submit_avgpass"):
        a, b = p1["R0_epi"][k], p1["R1_gigpo"][k]
        if a is not None and b is not None and b > a:
            repair_improved.append((k, a, b))
    if p2:
        fa = (p2.get("R0_epi") or {}).get("branch_F_first_failed")
        fb = (p2.get("R1_gigpo") or {}).get("branch_F_first_failed")
        if fa and fb and fb["delta"] > fa["delta"]:
            repair_improved.append(("F_branch_delta", fa["delta"], fb["delta"]))
    main_pass = d >= 0.005
    alt_pass = d >= -0.005 and bool(repair_improved)
    gate = d >= 0
    v = {
        "avg_pass_delta_R1_minus_R0": round(d, 4),
        "main_criterion(>=+0.005)": main_pass,
        "alt_criterion(>=-0.005 & repair improved)": alt_pass,
        "repair_metrics_strictly_improved": repair_improved,
        "direction_gate(R1-R0>=0 -> Stage2)": gate,
        # 口径说明：方向门（计划预注册原文"R1−R0 ≥ 0 才进 Stage 2，否则止损"）
        # 管"是否继续投入"；主/备判据管"能否宣称机制起效"。两者独立评判。
        "verdict": ("PASS-main" if main_pass else
                    "PASS-alt" if alt_pass else
                    "GATE-PASS-NO-SIGNAL(Δ>=0 门通过进 Stage 2；但主/备判据未达，"
                    "现阶段不得宣称机制起效)" if gate else
                    "FAIL-stop_loss(Δ<0，按预注册止损)"),
    }
    print(f"\n=== 预注册判据判定 ===\n{json.dumps(v, ensure_ascii=False, indent=1)}")
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2", action="store_true")
    args = ap.parse_args()
    p1 = phase1_pair()
    diff = task_diff_pair()
    p2 = phase2_pair() if args.phase2 else None
    v = verdict(p1, diff, p2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"phase1": p1, "task_diff": diff, "phase2": p2, "verdict": v},
                  f, ensure_ascii=False, indent=1)
    print(f"saved -> {OUT}")
