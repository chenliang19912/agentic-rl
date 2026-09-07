"""WS-2 Stage 2 分析驱动：gigpo 全量 4 iters vs 历史 epi 4 iters 配对对比。

口径完全复用 poc_gigpo_compare.py（其内部复用 c4_deepdive），只把 PAIR
按 iter 逐对参数化：epi_iter{i} = 历史 c4_train.py（eager 引擎采样）贪心评测，
gigpo_iter{i} = c4_train_gigpo.py（WS-1 胜出配置采样）贪心评测。

预注册判据（计划文件锁定于 Stage 2 开跑前，机械判定不掺主观）：
  成功：任一 gigpo iter 贪心 avg_pass ≥ 0.7526（历史峰值 0.7516 + 0.001 容差），
        或 存在 iter ≥ 0.7416（非劣 = 峰值 −0.01）且该 iter 贪心修复成功率 > 0%
        （哪怕 1/15）且该 iter F 分支 Δ > 0.17（Phase2 首跑隐藏判分口径）。
  失败：以上均不满足 → 如实写负结果（预置写法见 SCALE_UP §6.4）。

可比性口径（如实）：训练 rollout 引擎不同（gigpo 用 graph+prefixON 采样，
epi 历史用 eager；temp1.0 下轨迹=不同 RNG 实现，属重跑级噪声）；贪心评测
两侧同引擎同 CLI（A1 已证贪心跨引擎配置 0/89 翻转）。

用法：
  python3 poc_gigpo_stage2_analyze.py                # Phase1 全表（宿主机，纯 stdlib）
  python3 poc_gigpo_stage2_analyze.py --phase2 1     # 指定 iter 的 Phase2（容器内沙盒）
产物：logs/poc_gigpo/stage2_analyze.json / stage2_p2_rows_iter{i}.jsonl
"""
import argparse
import json
import os
import shutil

import poc_gigpo_compare as pgc

EPI = {i: f"logs/c4_grpo_iter{i}_multiturn.json" for i in range(4)}
GIGPO = {i: f"logs/poc_gigpo/stage2_eval_iter{i}.json" for i in range(4)}
OUT = "logs/poc_gigpo/stage2_analyze.json"
ROWS_SRC = "logs/poc_gigpo/stage15_p2_rows.jsonl"  # pgc.phase2_pair 固定写此处


def set_pair(i):
    pgc.PAIR = [(f"epi_iter{i}", EPI[i]), (f"gigpo_iter{i}", GIGPO[i])]


def phase1_all():
    report = {}
    print("=" * 70)
    for i in range(4):
        print(f"\n########## iter{i}: epi vs gigpo ##########")
        set_pair(i)
        p1 = pgc.phase1_pair()
        diff = pgc.task_diff_pair()
        report[f"iter{i}"] = {"phase1": p1, "task_diff": diff}
    print("\n" + "=" * 70)
    rows = []
    for i in range(4):
        g = report[f"iter{i}"]["phase1"][f"gigpo_iter{i}"]
        e = report[f"iter{i}"]["phase1"][f"epi_iter{i}"]
        rows.append((i, e["avg_pass"], g["avg_pass"],
                     g["repair_success_full"], g["visible_fixed_rate"]))
    print("iter | epi_avg | gigpo_avg | Δ      | gigpo修复率 | gigpo可见修复")
    for i, ea, ga, rs, vf in rows:
        print(f"{i}    | {ea:.4f}  | {ga:.4f}    | {ga-ea:+.4f} | {rs}     | {vf}")
    # 预注册判定（Phase1 部分；F 分支 Δ 需 --phase2）
    best = max(rows, key=lambda r: r[2])
    peak_pass = max(r[2] for r in rows)
    main = peak_pass >= 0.7526
    noninf = [r for r in rows if r[2] >= 0.7416 and (r[3] or 0) > 0]
    print(f"\n[预注册-主判据] 任一 iter ≥0.7526：{'PASS' if main else 'FAIL'}（峰值 {peak_pass:.4f} @ iter{best[0]}）")
    print(f"[预注册-备选] ≥0.7416 且贪心修复率>0% 的 iter：{[r[0] for r in noninf] or '无'}"
          f"（还需这些 iter 的 F 分支 Δ>0.17 才算成功，见 --phase2）")
    report["prereg_phase1"] = {
        "peak_gigpo_avg_pass": peak_pass, "main_criterion_ge_0.7526": main,
        "noninferior_iters_with_repair_gt0": [r[0] for r in noninf],
    }
    json.dump(report, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n[written] {OUT}")
    return report


def phase2_one(i):
    set_pair(i)
    s = pgc.phase2_pair()
    dst = f"logs/poc_gigpo/stage2_p2_rows_iter{i}.jsonl"
    if os.path.exists(ROWS_SRC):
        shutil.move(ROWS_SRC, dst)
        print(f"[rows] {dst}")
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out[f"iter{i}"]["phase2"] = s
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    fe = (s.get(f"epi_iter{i}") or {}).get("branch_F_first_failed")
    fg = (s.get(f"gigpo_iter{i}") or {}).get("branch_F_first_failed")
    print(f"\n[F 分支 Δ] epi={fe['delta'] if fe else None} gigpo={fg['delta'] if fg else None}"
          f"（预注册成功线 >0.17）")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2", type=int, default=-1, help="指定 iter 跑 Phase2；-1=不跑")
    a = ap.parse_args()
    phase1_all()
    if a.phase2 >= 0:
        phase2_one(a.phase2)
