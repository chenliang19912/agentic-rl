"""WS-2 PoC-B CPU/单卡单测：c4_gigpo + encode_chat_assistant_turns + 恒等性

在容器内跑（需 transformers/torch/numpy + checkpoints/c4_rft_merged 的 tokenizer）：
  python -u poc_gigpo_test.py
覆盖（预注册计划 WS-2 验证节）：
  T1 合成数据：mixed/饱和组的 A_epi/A_turn 手算对照；ω=0 退化恒等；γ=1.0 饱和组新增信号=0
  T2 verdicts/turn_anchors/classify_repair_turns：P/F/E 序列、锚点前缀、角色标注、
     超限 run 轮（无反馈）边界
  T3 真实 tokenizer + iter0 rollouts 20 条：labels 与 encode_chat_assistant_only 逐位一致、
     Σspans token 数 == labels 非 -100 数、span j 递增且与 assistant 消息数对齐
  T4 截断：小 max_len 下 span/labels 同步裁剪
  T5 loss 恒等性：turn 加权路径在均匀优势下 == 原标量路径（多形状、正负 c）
  T6 配对构造保证：γ=1.0 下 gigpo 与 epi 的零优势过滤集合在真实 iter0 上完全一致
  T7 环境自检：容器内 EPI_SOURCE 必须是 m3_env（防退化路径混入训练）
"""
import json
import sys

import torch

from c4_gigpo import (EPI_SOURCE, gigpo_advantages, turn_anchors,
                      verdicts_from_messages, classify_repair_turns)
from mask_utils import encode_chat_assistant_only, encode_chat_assistant_turns

PASS = []


def ok(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        raise SystemExit(f"TEST FAILED: {name} {detail}")


def synth_traj(task_id, verdicts, reward, final="answer"):
    """构造合成轨迹：verdicts 为每次 run 的可见判定；final=answer/run_over。"""
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "Task: fix it"}]
    for i, v in enumerate(verdicts):
        msgs.append({"role": "assistant", "content": f"<run_code>v{i}</run_code>"})
        out = "VISIBLE_TEST: PASSED" if v == "P" else (
            "VISIBLE_TEST: FAILED" if v == "F" else "Traceback: boom")
        msgs.append({"role": "user", "content": f"Execution result:\n{out}"})
    if final == "answer":
        msgs.append({"role": "assistant", "content": "<answer>def f(): pass</answer>"})
    else:  # 超限 run 轮：无反馈 user 消息
        msgs.append({"role": "assistant", "content": "<run_code>last</run_code>"})
    return {"task_id": task_id, "messages": msgs, "reward": reward,
            "r_pass": reward, "n_runs": len(verdicts)}


def t1_t2():
    # T2 基础：判定序列与锚点
    tr = synth_traj("A", ["F", "P"], 1.2)
    ok("T2.verdicts", verdicts_from_messages(tr["messages"]) == ["F", "P"])
    anchors = turn_anchors(tr["messages"])
    # assistant 轮：run0(前缀空) run1(前缀F) answer(前缀FP)
    ok("T2.anchors", [a[1] for a in anchors] == [(), ("F",), ("F", "P")],
       str([a[1] for a in anchors]))
    ok("T2.is_run", [a[2] for a in anchors] == [True, True, False])
    roles = classify_repair_turns(tr["messages"])
    ok("T2.roles", roles == {0: "fail", 1: "repair"}, str(roles))
    # 超限 run 轮（无反馈）
    tr2 = synth_traj("B", ["F"], 0.0, final="run_over")
    a2 = turn_anchors(tr2["messages"])
    ok("T2.over_limit_anchor", a2[-1][1] == ("F",) and a2[-1][2], str(a2))
    ok("T2.over_limit_role_absent", 1 not in classify_repair_turns(tr2["messages"]))
    # keep 角色：首跑即 P
    tr3 = synth_traj("C", ["P"], 1.2)
    ok("T2.role_keep", classify_repair_turns(tr3["messages"]) == {0: "keep"})

    # T1 合成组：task A mixed [1.2,0.0,1.2,0.0]、task B 饱和 [1.2]*4（G=4）。
    # B 组刻意造长度差（2 条 1-run、2 条 2-run）：γ<1 时同锚点桶 RTG 随剩余步数
    # 不同 → 同分组出现"效率信号"（Stage 0 的 115 turns 新发现之最小复现）；
    # γ=1.0 时桶内 RTG 恒同 → 新增信号=0（数学必然之最小复现）。
    gA = [synth_traj("A", ["F", "P"], 1.2), synth_traj("A", ["F", "F"], 0.0),
          synth_traj("A", ["P"], 1.2), synth_traj("A", ["F"], 0.0)]
    gB = [synth_traj("B", ["P"], 1.2), synth_traj("B", ["P"], 1.2),
          synth_traj("B", ["P", "P"], 1.2), synth_traj("B", ["P", "P"], 1.2)]
    rows = gA + gB
    A_turns, a_epi, info = gigpo_advantages(rows, 4, gamma=1.0, omega=1.0)
    # [1.2,0,1.2,0]: mean 0.6, pop-std 0.6 → adv = ±0.999998
    ok("T1.epi_mixed", abs(a_epi[0] - 1.0) < 1e-4, f"a_epi[0]={a_epi[0]:.4f}")
    ok("T1.epi_saturated_zero", all(abs(x) < 1e-9 for x in a_epi[4:8]))
    ok("T1.gamma1_no_new_signal",
       all(abs(t) < 1e-9 for i in range(4, 8) for t in A_turns[i]),
       "饱和组 γ=1.0 下 A_turn 全 0")
    # γ=0.9 时饱和组出现效率信号（不同长度 → RTG 不同）——Stage 0 发现的单测化
    A_g09, _ae09, _i09 = gigpo_advantages(rows, 4, gamma=0.9, omega=1.0)
    ok("T1.gamma09_efficiency_signal",
       any(abs(t) > 1e-6 for i in range(4, 8) for t in A_g09[i]),
       "γ<1 在同分组产生长度效率信号（预注册修订依据）")
    # ω=0 退化恒等
    A_w0, ae_w0, _ = gigpo_advantages(rows, 4, gamma=1.0, omega=0.0)
    ok("T1.omega0_identity",
       all(abs(A_w0[i][j] - ae_w0[i]) < 1e-9
           for i in range(8) for j in range(len(A_w0[i]))))
    # mixed 组内信用重分配存在：同轨迹不同 turn 的 A_turn 不全相等
    ok("T1.mixed_reallocation",
       any(len(set(round(x, 6) for x in A_turns[i])) > 1 for i in range(4)))
    ok("T1.info_fields", {"n_anchor_groups", "frac_anchor_singleton",
                          "mean_abs_a_step"} <= set(info.keys()))


def t3_t4_t6(tok_path="checkpoints/c4_rft_merged",
             rollouts="checkpoints/c4_grpo/iter0_rollouts.jsonl"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    rows = [json.loads(l) for l in open(rollouts)][:20]
    n_cover = n_lab_eq = n_jmax = 0
    for r in rows:
        ids0, _, lab0 = encode_chat_assistant_only(tok, r["messages"], max_len=4096)
        ids1, _, lab1, spans = encode_chat_assistant_turns(tok, r["messages"], max_len=4096)
        n_lab_eq += (ids0 == ids1 and lab0 == lab1)
        span_tok = sum(e - s for (s, e, _j) in spans)
        n_cover += (span_tok == sum(1 for x in lab1 if x != -100))
        n_assist = sum(1 for m in r["messages"] if m["role"] == "assistant")
        n_jmax += (all(0 <= j < n_assist for (_s, _e, j) in spans)
                   and len(spans) <= n_assist
                   and [j for (_s, _e, j) in spans] == sorted(j for (_s, _e, j) in spans))
    ok("T3.labels_bitwise_equal", n_lab_eq == len(rows), f"{n_lab_eq}/{len(rows)}")
    ok("T3.span_coverage", n_cover == len(rows), f"{n_cover}/{len(rows)}")
    ok("T3.span_j_aligned", n_jmax == len(rows), f"{n_jmax}/{len(rows)}")

    # T4 截断：max_len=64
    r = rows[0]
    ids, _, lab, spans = encode_chat_assistant_turns(tok, r["messages"], max_len=64)
    ok("T4.trunc_len", len(ids) <= 64 and len(lab) == len(ids))
    ok("T4.trunc_spans_in_range", all(0 <= s < e <= len(ids) for (s, e, _j) in spans))
    ok("T4.trunc_coverage", sum(e - s for (s, e, _j) in spans)
       == sum(1 for x in lab if x != -100))

    # T6 配对构造保证：γ=1.0 下过滤集合与 epi 一致（全量 iter0）
    all_rows = [json.loads(l) for l in open(rollouts)]
    A_turns, a_epi, _ = gigpo_advantages(all_rows, 4, gamma=1.0, omega=1.0)
    set_g = set(i for i in range(len(all_rows)) if any(abs(x) > 1e-4 for x in A_turns[i]))
    set_e = set(i for i in range(len(all_rows)) if abs(a_epi[i]) > 1e-4)
    ok("T6.filter_set_identical", set_g == set_e,
       f"|gigpo|={len(set_g)} |epi|={len(set_e)}")


def t5():
    from c4_train_gigpo import seq_logp, turn_weighted_logp
    torch.manual_seed(1)
    for (B, T, V) in ((1, 17, 29), (1, 5, 11), (2, 33, 50)):
        logits = torch.randn(B, T, V)
        ids = torch.randint(0, V, (B, T))
        mask = torch.zeros(B, T)
        mask[:, 2:T // 2] = 1.0
        for c in (1.0, -0.7, 2.31):
            a_tok = torch.full((B, T), c)
            ref = -(c * seq_logp(logits, ids, mask))
            new = -turn_weighted_logp(logits, ids, mask, a_tok)
            d = float((ref - new).abs().max())
            ok(f"T5.identity_{B}x{T}x{V}_c{c}", d < 1e-5, f"diff={d:.2e}")


def t7():
    ok("T7.epi_source_m3_env", EPI_SOURCE == "m3_env.grpo_advantages",
       f"EPI_SOURCE={EPI_SOURCE}")


if __name__ == "__main__":
    t1_t2()
    t5()
    t7()
    t3_t4_t6()
    print(f"\nALL {len(PASS)} TESTS PASS")
