"""PoC-B Stage 0（WS-2）：GiGPO turn 级优势的离线反事实正式化报告（0 GPU）

在已落盘的 checkpoints/c4_grpo/iter{0..3}_rollouts.jsonl（各 89 题 × G4 = 356 条，
含 messages/reward/grade 全字段）上正式化预注册所需的全部反事实数字：

  1. 组构成：mixed / 全对饱和 / 全零 / 中间同分 —— "零优势过滤滤掉的是什么"
  2. 采样层修复存在性（mixed 组）：首跑失败轨迹 → 可见修复率 → 终版满分率
  3. 锚点密度：全前缀锚 vs 粗锚 (k,last)（粗锚仅敏感性对照，不进训练臂）
  4. 新增信号校验：|A_epi|=0 且 |A_step|>0 的 turn 数（预注册预期 = 0，数学必然）
  5. 重分配幅度：mixed 组内非零 A_epi turn 中 |A_turn−A_epi| > 0.3|A_epi| 占比
  6. 修复 turn 信用锐化：repair/fail/keep/answer 各角色的 mean(A_turn−A_epi)
     （预期方向：repair > 0、fail < 0 —— GiGPO 把正优势集中到修复 turn）
  7. γ∈{0.9,0.95,1.0}×ω∈{0.5,1.0} 网格（训练臂固定 γ=0.95/ω=1.0）
  8. 量纲比 mean|A_turn|/mean|A_epi|（>1.5 预警线，见计划风险表）

宿主机可直接跑（python3 poc_gigpo_offline.py）；c4_gigpo 在无 numpy 环境
自动退化纯 python 等价实现（EPI_SOURCE 自证）。产物 logs/poc_gigpo/stage0_report.json。
"""
import json
import os
from collections import Counter, defaultdict

from c4_gigpo import (gigpo_advantages, turn_anchors, coarse_anchor,
                      classify_repair_turns, verdicts_from_messages, EPI_SOURCE)

G = 4
ITERS = [0, 1, 2, 3]
OUT = "logs/poc_gigpo/stage0_report.json"


def load_iter(it):
    path = f"checkpoints/c4_grpo/iter{it}_rollouts.jsonl"
    return [json.loads(l) for l in open(path)]


def task_groups(rows):
    """按落盘顺序切 G 条一组（已验证 task-major），返回 [(task_id, [rows])]."""
    groups = []
    for i in range(0, len(rows), G):
        g = rows[i:i + G]
        assert len(set(r["task_id"] for r in g)) == 1, f"group {i} 跨任务"
        groups.append((g[0]["task_id"], g))
    return groups


def group_composition(rows):
    comp = Counter()
    mixed_tasks = []
    for tid, g in task_groups(rows):
        rs = [r["reward"] for r in g]
        if max(rs) - min(rs) > 1e-9:
            comp["mixed"] += 1
            mixed_tasks.append(tid)
        elif all((r["grade"] or {}).get("pass_rate") == 1.0 for r in g):
            comp["same_saturated_fullpass"] += 1
        elif all(((r["grade"] or {}).get("pass_rate") or 0.0) == 0.0 for r in g):
            comp["same_all_zeropass"] += 1
        else:
            comp["same_partial"] += 1
    return dict(comp), set(mixed_tasks)


def repair_presence(rows, mixed_tasks):
    """mixed 组内：首跑失败轨迹的采样层修复存在性（预注册叙事的事实基础）。"""
    n_first_fail = n_vis_repair = n_final_full = 0
    for r in rows:
        if r["task_id"] not in mixed_tasks:
            continue
        seq = verdicts_from_messages(r["messages"])
        if not seq or seq[0] not in ("F", "E"):
            continue
        n_first_fail += 1
        bad = next((i for i, x in enumerate(seq) if x in ("F", "E")), None)
        if bad is not None and "P" in seq[bad + 1:]:
            n_vis_repair += 1
        if (r["grade"] or {}).get("pass_rate") == 1.0:
            n_final_full += 1
    return {"mixed_first_fail_traj": n_first_fail,
            "visible_repair": n_vis_repair,
            "visible_repair_pct": round(100.0 * n_vis_repair / max(n_first_fail, 1), 1),
            "final_fullpass": n_final_full,
            "final_fullpass_pct": round(100.0 * n_final_full / max(n_first_fail, 1), 1)}


def anchor_density(rows):
    """全前缀锚 vs 粗锚的桶密度与桶内 reward 方差占比。"""
    out = {}
    for name, fn in (("full_prefix", turn_anchors), ("coarse_k_last", coarse_anchor)):
        buckets = defaultdict(list)   # key -> [(reward, 轨迹下标)]
        for i, r in enumerate(rows):
            for (_j, p, _ir, _k) in fn(r["messages"]):
                buckets[(r["task_id"], p)].append((r["reward"], i))
        sizes = Counter(len(v) for v in buckets.values())
        ge2 = sum(c for s, c in sizes.items() if s >= 2)
        var = sum(1 for v in buckets.values()
                  if len(v) >= 2 and max(x[0] for x in v) - min(x[0] for x in v) > 1e-9)
        # 跨轨迹口径：桶内 ≥2 条**不同轨迹**才算可分组（同轨迹自配对无信息，
        # 如超限 run 轮与 answer 轮共享全串锚）
        ge2x = sum(1 for v in buckets.values() if len(set(x[1] for x in v)) >= 2)
        varx = sum(1 for v in buckets.values()
                   if len(set(x[1] for x in v)) >= 2
                   and max(x[0] for x in v) - min(x[0] for x in v) > 1e-9)
        out[name] = {"n_buckets": len(buckets),
                     "size_hist": dict(sorted(sizes.items())),
                     "frac_size_ge2": round(ge2 / max(len(buckets), 1), 4),
                     "frac_with_reward_var": round(var / max(len(buckets), 1), 4),
                     "frac_size_ge2_cross_traj": round(ge2x / max(len(buckets), 1), 4),
                     "frac_with_var_cross_traj": round(varx / max(len(buckets), 1), 4)}
    return out


def per_turn_tables(rows, gamma, omega):
    """返回逐 turn 表：[(task, j, role, a_epi, a_turn, in_mixed)] + info。"""
    A_turns, a_epi, info = gigpo_advantages(rows, G, gamma=gamma, omega=1.0)
    _mixed = set(group_composition(rows)[1])
    tab = []
    for i, r in enumerate(rows):
        anchors = turn_anchors(r["messages"])
        roles = classify_repair_turns(r["messages"])
        for (j, _p, is_run, _k) in anchors:
            a_step = (A_turns[i][j] - a_epi[i])          # ω=1 口径下的 A_step
            a_turn = a_epi[i] + omega * a_step
            role = roles.get(j, "answer" if not is_run else "run_over_limit")
            tab.append((r["task_id"], j, role, a_epi[i], a_turn, r["task_id"] in _mixed))
    return tab, a_epi, info


def signal_and_sharpening(tab):
    new_signal = sum(1 for (_t, _j, _ro, ae, at, _m) in tab
                     if abs(ae) <= 1e-4 and abs(at - ae) > 1e-4)
    mixed_nz = [(ae, at) for (_t, _j, _ro, ae, at, m) in tab if m and abs(ae) > 1e-4]
    realloc = sum(1 for ae, at in mixed_nz if abs(at - ae) > 0.3 * abs(ae))
    mag_ratio = (sum(abs(at) for _t, _j, _ro, _ae, at, _m in tab) /
                 max(sum(abs(ae) for _t, _j, _ro, ae, _at, _m in tab), 1e-9))
    by_role = defaultdict(list)
    for (_t, _j, role, ae, at, m) in tab:
        if m:
            by_role[role].append(at - ae)
    sharpen = {role: {"n": len(v),
                      "mean_delta_A": round(sum(v) / len(v), 4) if v else None}
               for role, v in sorted(by_role.items())}
    return {"new_signal_turns(expected 0)": new_signal,
            "mixed_nonzero_epi_turns": len(mixed_nz),
            "realloc_frac_gt_0.3": round(realloc / max(len(mixed_nz), 1), 4),
            "magnitude_ratio_meanAbsAturn_over_Aepi": round(mag_ratio, 4),
            "sharpening_by_role_mixed_only": sharpen}


def main():
    report = {"epi_source": EPI_SOURCE, "G": G, "iters": {}}
    for it in ITERS:
        rows = load_iter(it)
        comp, mixed = group_composition(rows)
        tab, _ae, info = per_turn_tables(rows, gamma=0.95, omega=1.0)
        report["iters"][f"iter{it}"] = {
            "n_traj": len(rows),
            "group_composition": comp,
            "repair_presence_mixed": repair_presence(rows, mixed),
            "anchor_density": anchor_density(rows),
            "gigpo_info": info,
            "signal_sharpening_g0.95_w1.0": signal_and_sharpening(tab),
        }
        print(f"--- iter{it} ---")
        print(json.dumps(report["iters"][f"iter{it}"]["group_composition"],
                         ensure_ascii=False))
        print(json.dumps(report["iters"][f"iter{it}"]["repair_presence_mixed"],
                         ensure_ascii=False))
        print(json.dumps(report["iters"][f"iter{it}"]["signal_sharpening_g0.95_w1.0"],
                         ensure_ascii=False, indent=1))

    # γ/ω 网格（iter0，训练臂固定 γ=0.95/ω=1.0；网格仅离线扫）
    rows0 = load_iter(0)
    grid = {}
    for gamma in (0.9, 0.95, 1.0):
        tab_g, _ae, _info = per_turn_tables(rows0, gamma=gamma, omega=1.0)
        for omega in (0.5, 1.0):
            # A_step 与 ω 无关：从 ω=1 表重构 A_turn(ω) = a_epi + ω·(A_turn(1)−a_epi)
            re_tab = [(t, j, ro, ae, ae + omega * (at - ae), m)
                      for (t, j, ro, ae, at, m) in tab_g]
            s = signal_and_sharpening(re_tab)
            grid[f"gamma{gamma}_omega{omega}"] = {
                k: s[k] for k in ("realloc_frac_gt_0.3",
                                  "magnitude_ratio_meanAbsAturn_over_Aepi",
                                  "new_signal_turns(expected 0)")}
            grid[f"gamma{gamma}_omega{omega}"]["sharpen_repair_fail"] = {
                r: s["sharpening_by_role_mixed_only"].get(r, {}).get("mean_delta_A")
                for r in ("repair", "fail")}
    report["grid_iter0"] = grid
    print("--- γ/ω 网格（iter0）---")
    print(json.dumps(grid, ensure_ascii=False, indent=1))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
