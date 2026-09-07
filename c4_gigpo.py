"""GiGPO 式二级分组优势（episode 级 + anchor-step 级）—— WS-2 PoC-B 规范实现

设计（预注册于 2026-09-06 迭代计划，判据见计划 WS-2 节）：
- 锚点 = (task_id, 该 turn 之前的可见判定 P/F/E 全前缀)。
  **禁止用原始 feedback 文本做锚**——其中含各轨迹自己的代码 stdout，
  跨轨迹必然 100% singleton → 零信号。
- RTG_t = γ^(T-1-t) · R：终局标量 reward 折扣回传（γ=0.95 训练臂固定；
  轨迹仅 1-6 步，γ^5≈0.77，温和 recency 加权，贴近 GiGPO 在 ALFWorld 的量级）。
  **奖励函数一字不改**（单变量纪律：只动优势估计器）。
- A_step = 锚点桶内 RTG 归一化（population std，同 grpo_advantages 口径）；
  singleton 桶 A_step = 0。
- A_turn[j] = A_epi + ω·A_step(j)（ω=1.0；两项各自组内归一后相加，
  GiGPO 原式，arXiv 2505.10978）。
- A_epi 复用 m3_env.grpo_advantages（零改动）；宿主机无 numpy/vllm 时
  退化为纯 python 等价实现（仅供 Stage 0 离线分析；训练路径必走 m3_env，
  由 EPI_SOURCE 字段自证走的是哪条）。

Stage 0 反事实实测依据（iter0-3 rollouts，重塑了预注册假设）：
- all-same 组的锚点子组继承零方差 → **GiGPO 新增信号 = 0 是数学必然**；
  真实机理 = mixed 组内**信用重分配**（把正优势从"修复轨迹全程（含失败的
  首跑 turn）"集中到"失败之后的修复 turn"），不是解锁全败组。
- 修复行为在采样层存在（temp1.0 mixed 组首跑失败轨迹 ~31% 可见修复）；
  贪心修复率 0% 的病灶 = 贪心策略不选择修复 → 信用锐化对准的正是这个病灶。
"""
import re

VIS_RE = re.compile(r"VISIBLE_TEST:\s*(PASSED|FAILED)")

try:
    from m3_env import grpo_advantages as _epi_adv
    EPI_SOURCE = "m3_env.grpo_advantages"
except ImportError:          # 宿主机（py3.6 无 numpy/vllm）离线分析退化路径
    EPI_SOURCE = "pure-python-fallback(offline-only)"

    def _epi_adv(rewards, group_size, eps=1e-6):
        adv = [0.0] * len(rewards)
        for i in range(0, len(rewards), group_size):
            g = rewards[i:i + group_size]
            mu = sum(g) / len(g)
            std = (sum((x - mu) ** 2 for x in g) / len(g)) ** 0.5
            for j, x in enumerate(g):
                adv[i + j] = (x - mu) / (std + eps)
        return adv


def verdicts_from_messages(messages):
    """环境反馈 user 消息 → 'P'/'F'/'E' 序列（与 c4_deepdive.run_feedback_seq 同口径）"""
    seq = []
    for m in messages:
        if m["role"] != "user" or not m["content"].startswith("Execution result:"):
            continue
        found = VIS_RE.findall(m["content"])
        seq.append(found[0][0] if found else "E")
    return seq


def turn_anchors(messages):
    """assistant 轮结构：[(j, 前置判定前缀 tuple, is_run, run序号k或None)]

    第 r 个（0-index）run 轮的前置状态 = verdicts[:r]（本轮执行结果尚未发生）；
    answer/invalid 轮前置状态 = 全部判定串；MAX_RUNS 超限的 run 轮无反馈，
    verdicts[:r] 自然等于全串。
    """
    verdicts = verdicts_from_messages(messages)
    out = []
    run_idx = 0
    j = 0
    for m in messages:
        if m["role"] != "assistant":
            continue
        if "<run_code>" in m["content"]:
            out.append((j, tuple(verdicts[:run_idx]), True, run_idx))
            run_idx += 1
        else:
            out.append((j, tuple(verdicts), False, None))
        j += 1
    return out


def gigpo_advantages(results, group_size, gamma=0.95, omega=1.0, eps=1e-6):
    """返回 (A_turns, a_epi, info)。

    A_turns[i] = 轨迹 i 的 per-turn 优势 list（按 assistant 消息顺序，
                 与 turn_anchors 的 j 对齐）；
    a_epi[i]   = 轨迹级优势（复用 grpo_advantages，对照/恒等性自检用）；
    info       = 归因统计（进 stats.jsonl 的预注册字段）。
    """
    rewards = [r["reward"] for r in results]
    a_epi = list(_epi_adv(rewards, group_size))

    # 1) 收集全部 turn：(轨迹i, 轮j, 锚点, RTG)
    turns = []
    per_traj_anchors = []
    for i, r in enumerate(results):
        anchors = turn_anchors(r["messages"])
        per_traj_anchors.append(anchors)
        T = len(anchors)
        R = r["reward"]
        for (j, prefix, _is_run, _k) in anchors:
            rtg = (gamma ** (T - 1 - j)) * R
            turns.append((i, j, (r["task_id"], prefix), rtg))

    # 2) 锚点桶内归一化 → A_step（singleton 桶 = 0）
    buckets = {}
    for t in turns:
        buckets.setdefault(t[2], []).append(t)
    a_step = {}
    n_single = 0
    n_var = 0
    for key, members in buckets.items():
        if len(members) < 2:
            n_single += 1
            for (i, j, _a, _r) in members:
                a_step[(i, j)] = 0.0
            continue
        vals = [m[3] for m in members]
        mu = sum(vals) / len(vals)
        std = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
        if std > eps:
            n_var += 1
        for (i, j, _a, rtg) in members:
            a_step[(i, j)] = (rtg - mu) / (std + eps)

    # 3) 合成 per-turn 优势
    A_turns = []
    for i in range(len(results)):
        A_turns.append([a_epi[i] + omega * a_step[(i, j)]
                        for (j, _p, _r, _k) in per_traj_anchors[i]])

    n_turns = len(turns)
    nz_turn = sum(1 for v in A_turns for x in v if abs(x) > 1e-4)
    info = {
        "epi_source": EPI_SOURCE,
        "gamma": gamma, "omega": omega,
        "n_turns_total": n_turns,
        "n_anchor_groups": len(buckets),
        "n_anchor_singleton": n_single,
        "frac_anchor_singleton": round(n_single / max(len(buckets), 1), 4),
        "n_anchor_with_var": n_var,
        "mean_abs_a_step": round(sum(abs(v) for v in a_step.values()) / max(n_turns, 1), 4),
        "frac_turns_nonzero": round(nz_turn / max(n_turns, 1), 4),
    }
    return A_turns, a_epi, info


# ---- 以下工具供 Stage 0 离线分析与单测复用 ----

def coarse_anchor(messages):
    """粗锚点 (已执行次数k, 最后判定) —— 仅离线敏感性对照，不进训练臂。
    返回形状与 turn_anchors 一致：[(j, key, is_run, k)]。"""
    verdicts = verdicts_from_messages(messages)
    out = []
    run_idx = 0
    j = 0
    for m in messages:
        if m["role"] != "assistant":
            continue
        if "<run_code>" in m["content"]:
            out.append((j, (run_idx, verdicts[run_idx - 1] if run_idx else None),
                        True, run_idx))
            run_idx += 1
        else:
            out.append((j, (run_idx, verdicts[-1] if verdicts else None),
                        False, None))
        j += 1
    return out


def classify_repair_turns(messages):
    """标注每条轨迹的 run 轮角色：
    'repair' = 该轮执行后判定翻到 P 且此前出现过 F/E；
    'fail'   = 该轮执行后判定为 F/E；
    'keep'   = 该轮执行后判定为 P 且此前无 F/E（含首跑即 P 的验证轮）。
    返回 {turn_j: role}。"""
    verdicts = verdicts_from_messages(messages)
    roles = {}
    run_idx = 0
    j = 0
    for m in messages:
        if m["role"] != "assistant":
            continue
        if "<run_code>" in m["content"]:
            if run_idx < len(verdicts):
                v = verdicts[run_idx]
                prior = verdicts[:run_idx]
                if v == "P" and any(x in ("F", "E") for x in prior):
                    roles[j] = "repair"
                elif v in ("F", "E"):
                    roles[j] = "fail"
                else:
                    roles[j] = "keep"
            run_idx += 1
        j += 1
    return roles
