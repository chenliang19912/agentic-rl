"""C4 WS-2 PoC-B：turn 级优势训练循环（c4_train.py 的姊妹拷贝，沿用
m3_train_reinforce.py 的先例——基线文件保持逐 bit 可复现，不加分支 flag）。

与 c4_train.py 的差异（单变量纪律：奖励函数/数据/超参全不动，只动优势估计器）：
  1. --adv {epi,gigpo}：epi = 与 c4_train.py 同路径（轨迹级标量优势，R0 臂）；
     gigpo = c4_gigpo.gigpo_advantages 二级分组（episode + anchor-step，R1 臂）
  2. --replay <jsonl>：跳过 rollout，直接消费历史 iter rollouts（配对重放训练；
     R0/R1 同 --seed 同子集 → 完美单变量对比）
  3. 显式 --seed：random.shuffle 与 LoRA 初始化都可复现（原 c4_train 未定种）
  4. gigpo 路径 loss 改为 per-turn token 加权：
       lp = Σ(tok_logp·mask·A_tok)/Σmask，A_tok[span_j] = A_turn[j]
     恒等性断言（启动时 CPU 自检）：A_turn 全等于标量 c 时与
     -(c·seq_logp) 差 < 1e-6 —— replay 配对可比的前提
  5. KL 保持整轨迹 seq 级 k3 原样（与优势无关的正则，不动）
  6. stats 增加 turn 级归因字段（frac_turns_nonzero / mean_abs_A_step /
     n_anchor_groups / frac_anchor_singleton / magnitude_ratio / n_turns_no_span）

预注册（2026-09-06，Stage 0 反事实后修订，训练未开跑前锁定）：
  - 训练臂 γ=1.0、ω=1.0。修订依据：Stage 0 实测 γ<1 会在同分组引入
    **长度效率信号**（iter0 有 115 个 turn 在 A_epi=0 时获得非零 A_step，
    方向="同分但更短的轨迹更正"）——那是第二个作用机制，会污染
    "信用重分配→修复锐化"的单机制归因；γ=1.0 时新增信号=0（数学必然），
    纯重分配。γ=0.95 效率通道记录为 v2 候选（logs/poc_gigpo/stage0_report.json）。
  - γ=1.0 下零优势过滤集合与 epi 臂完全一致（组内同分 → 桶内 RTG 全同 →
    A_step=0），R0/R1 消费同一 192 条子集，配对性由构造保证。
  - Stage 1.5 主判据：R1−R0 贪心 avg_pass ≥ +0.005，或 ≥ −0.005 且修复类
    指标（repair_success / visible_fixed_rate / F 分支 Δ）任一严格改善。
  - R0 vs 历史 iter1（0.7516）单独报告为 replay 保真度（预期 ≤0.01，
    差异来源：shuffle 子集不可复原 + fp16 非确定性；不作复现主张）。

用法（docker 单卡）：
  # Stage 1.5 配对 replay（同 seed）
  python -u c4_train_gigpo.py --adv epi   --replay checkpoints/c4_grpo/iter0_rollouts.jsonl \
      --seed 42 --tag R0 --out checkpoints/c4_replay
  python -u c4_train_gigpo.py --adv gigpo --replay checkpoints/c4_grpo/iter0_rollouts.jsonl \
      --seed 42 --tag R1 --out checkpoints/c4_replay
  # Stage 2 全量（门通过后；--eager/--prefix 用 WS-1 胜出配置）
  python -u c4_train_gigpo.py --adv gigpo --n-iters 4 --seed 42 --out checkpoints/c4_gigpo
"""
import argparse
import gc
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3_env import grpo_advantages            # 环境无关，直接复用（检索 Agent 项目栈）
from c4_gigpo import gigpo_advantages
from c4_rollout import rollout, load_tasks
from mask_utils import encode_chat_assistant_only, encode_chat_assistant_turns

# ---- 超参（与 c4_train.py 逐项相同——单变量纪律）----
N_ITERS = 4
GROUP_SIZE = 4
LR = 5e-6
KL_BETA = 0.05
MAX_LEN = 4096
ACC = 8
MAX_EFFECTIVE = 192
TEMP = 1.0

LORA = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                  target_modules="all-linear", task_type="CAUSAL_LM")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adv", choices=["epi", "gigpo"], default="gigpo")
    ap.add_argument("--replay", default=None, help="历史 rollouts jsonl（跳过 rollout）")
    ap.add_argument("--tag", default="", help="checkpoint/日志后缀（如 R0/R1）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--base", default="checkpoints/c4_rft_merged")
    ap.add_argument("--out", default="checkpoints/c4_gigpo")
    ap.add_argument("--tasks", default="data/c4/tasks_train.jsonl")
    ap.add_argument("--hidden", default="data/c4/hidden_train.jsonl")
    ap.add_argument("--n-iters", type=int, default=N_ITERS)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--kl-beta", type=float, default=KL_BETA)
    ap.add_argument("--max-effective", type=int, default=MAX_EFFECTIVE)
    ap.add_argument("--gpu-util", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    # rollout 引擎配置（Stage 2 用 WS-1 胜出配置；默认 = 现状）
    ap.add_argument("--eager", type=int, default=1)
    ap.add_argument("--prefix", type=int, default=-1, help="-1=引擎默认 / 0=OFF / 1=ON")
    return ap.parse_args()


def seq_logp(logits, input_ids, mask):
    """assistant token 平均 logp（与 c4_train.py 逐字相同）"""
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tok_logp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:]
    return (tok_logp * m).sum(-1) / m.sum(-1).clamp(min=1)


def turn_weighted_logp(logits, input_ids, mask, a_tok):
    """per-turn 优势加权的 token 平均 logp。
    恒等性：a_tok 在 mask 上恒为 c 时 == c · seq_logp（启动自检断言）。"""
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tok_logp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:]
    return (tok_logp * m * a_tok[:, 1:]).sum(-1) / m.sum(-1).clamp(min=1)


def identity_selftest():
    """恒等性自检（CPU、随机张量）：turn 加权路径在均匀优势下 == 原路径。"""
    torch.manual_seed(0)
    B, T, V = 1, 17, 29
    logits = torch.randn(B, T, V)
    ids = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T)
    mask[0, 3:9] = 1.0
    mask[0, 12:15] = 1.0
    c = 1.37
    a_tok = torch.full((B, T), c)
    ref = -(c * seq_logp(logits, ids, mask))
    new = -turn_weighted_logp(logits, ids, mask, a_tok)
    d = abs(float(ref) - float(new))
    assert d < 1e-6, f"恒等性自检失败: diff={d}"
    print(f"identity selftest PASS (diff={d:.2e})", flush=True)


def encode_trajectory(tok, messages):
    ids, _, labels = encode_chat_assistant_only(tok, messages, max_len=MAX_LEN)
    mask = [1 if l != -100 else 0 for l in labels]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def encode_trajectory_turns(tok, messages):
    ids, _, labels, spans = encode_chat_assistant_turns(tok, messages, max_len=MAX_LEN)
    mask = [1 if l != -100 else 0 for l in labels]
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32), spans)


def build_llm_for_rollout(model_path, args):
    from vllm import LLM
    kw = dict(dtype="float16", gpu_memory_utilization=args.gpu_util,
              max_model_len=8192, enforce_eager=bool(args.eager))
    if args.prefix in (0, 1):
        kw["enable_prefix_caching"] = bool(args.prefix)
    t0 = time.perf_counter()
    llm = LLM(model=model_path, **kw)
    print(f"  [engine] built in {time.perf_counter()-t0:.1f}s kwargs={kw}", flush=True)
    return llm


def main():
    args = parse_args()
    identity_selftest()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    tasks, hidden = load_tasks(args.tasks, args.hidden)
    print(f"adv={args.adv} replay={args.replay} seed={args.seed} "
          f"gamma={args.gamma} omega={args.omega} base={args.base}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    merged_dir = args.base
    prev_fmt = 0.0
    n_iters = 1 if args.replay else args.n_iters

    for it in range(n_iters):
        print(f"\n===== ITER {it}{tag} =====", flush=True)
        t_iter = time.perf_counter()

        # ---- 1. rollout 或 replay ----
        if args.replay:
            results = [json.loads(l) for l in open(args.replay)]
            print(f"  replay: {len(results)} trajectories from {args.replay}", flush=True)
        else:
            llm = build_llm_for_rollout(merged_dir, args)
            results, llm = rollout(merged_dir, tasks, hidden, G=args.group_size,
                                   temperature=TEMP, max_new_tokens=args.max_new_tokens,
                                   llm=llm)
            del llm
            gc.collect()
            torch.cuda.empty_cache()

        # ---- 2. 优势 ----
        rewards = [r["reward"] for r in results]
        if args.adv == "gigpo":
            A_turns, a_epi, ginfo = gigpo_advantages(
                results, args.group_size, gamma=args.gamma, omega=args.omega)
        else:
            a_epi = grpo_advantages(rewards, args.group_size).tolist()
            A_turns = None
            ginfo = {"epi_source": "m3_env.grpo_advantages", "adv_mode": "epi"}
        stat = {
            "iter": it, "tag": args.tag, "adv": args.adv,
            "seed": args.seed, "gamma": args.gamma, "omega": args.omega,
            "replay": bool(args.replay),
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "avg_pass": round(sum(r["r_pass"] for r in results) / len(results), 4),
            "fmt_pct": round(sum(r["r_format"] for r in results) / len(results) * 100, 1),
            "submit_pct": round(sum(1 for r in results if r["answer_code"]) / len(results) * 100, 1),
            "avg_runs": round(sum(r["n_runs"] for r in results) / len(results), 2),
            "dup_cnt": sum(1 for r in results if (r["grade"] or {}).get("dup_ref")),
            "grade_timeout_cnt": sum(1 for r in results if (r["grade"] or {}).get("timed_out")),
            **{k: v for k, v in ginfo.items() if k != "epi_source"},
            "epi_source": ginfo.get("epi_source"),
        }
        print(json.dumps(stat, ensure_ascii=False), flush=True)

        if not args.replay:
            with open(f"{out_dir}/iter{it}{tag}_rollouts.jsonl", "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 坍缩熔断（口径与 c4_train.py 一致；replay 单轮不触发）
        if it > 0 and stat["fmt_pct"] < 0.5 * prev_fmt:
            print(f"ABORT: fmt_pct collapsed ({prev_fmt} -> {stat['fmt_pct']})", flush=True)
            return
        prev_fmt = stat["fmt_pct"]

        # ---- 3. HF + LoRA 策略更新（超参与 c4_train.py 逐项相同）----
        t_train = time.perf_counter()
        base = AutoModelForCausalLM.from_pretrained(
            merged_dir, torch_dtype=torch.float16).cuda()
        base.config.use_cache = False
        model = get_peft_model(base, LORA)
        model.gradient_checkpointing_enable()
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        if args.adv == "gigpo":
            # turn 级零优势过滤：全部 turn |A|≤1e-4 → 整条跳过（γ=1.0 下与
            # 轨迹级过滤集合恒等——组内同分 ⇒ 桶内 RTG 全同 ⇒ A_step=0）
            traj = [(r, A_turns[i], a_epi[i]) for i, r in enumerate(results)
                    if any(abs(x) > 1e-4 for x in A_turns[i])]
        else:
            traj = [(r, None, a) for (r, a) in zip(results, a_epi) if abs(a) > 1e-4]
        n_before = len(results)
        print(f"  effective trajectories: {len(traj)}/{n_before}", flush=True)
        if not traj:
            print("no non-zero advantage; skip update this iter", flush=True)
            stat["loss"] = 0.0
            with open(f"{out_dir}/stats{tag}.jsonl", "a") as f:
                f.write(json.dumps(stat, ensure_ascii=False) + "\n")
            continue
        random.shuffle(traj)               # seed 已显式固定（R0/R1 同子集）
        traj = traj[:args.max_effective]

        opt.zero_grad()
        n_acc, tot_loss = 0, 0.0
        n_turns_no_span = 0
        mag_num = mag_den = 0.0
        for step, (r, A_turn, a) in enumerate(traj):
            if args.adv == "gigpo":
                ids, mask, spans = encode_trajectory_turns(tok, r["messages"])
                if mask.sum() < 3:
                    continue
                span_js = set(j for (_s, _e, j) in spans)
                n_turns_no_span += sum(1 for j in range(len(A_turn)) if j not in span_js)
                a_tok = torch.zeros(len(ids), dtype=torch.float32)
                for (s, e, j) in spans:
                    if j < len(A_turn):
                        # float() 显式转 python 标量：A_turn[j] 是 numpy.float32，
                        # 直接赋给 torch.FloatTensor 切片会 TypeError（Stage1.5 首跑实测）
                        a_tok[s:e] = float(A_turn[j])
                if a_tok.abs().sum() == 0:
                    continue
                mag_num += float(a_tok.abs().sum())
                mag_den += float(mask.sum())
                ids_b = ids.unsqueeze(0).cuda()
                mask_b = mask.unsqueeze(0).cuda()
                a_b = a_tok.unsqueeze(0).cuda()
                out = model(input_ids=ids_b)
                lp = turn_weighted_logp(out.logits, ids_b, mask_b, a_b)
                # KL 用"无优势加权"的 seq 级 logp（k3 口径与 c4_train 逐字一致；
                # 均匀优势下 lp == c·lp_seq，整条 loss 退化为原路径——恒等性自检覆盖）
                lp_seq = seq_logp(out.logits, ids_b, mask_b)
                with torch.no_grad(), model.disable_adapter():
                    ref_out = model(input_ids=ids_b)
                    ref_lp = seq_logp(ref_out.logits, ids_b, mask_b).detach()
                kl = (torch.exp(ref_lp - lp_seq) - (ref_lp - lp_seq) - 1.0)
                loss = -lp + args.kl_beta * kl
            else:
                ids, mask = encode_trajectory(tok, r["messages"])
                if mask.sum() < 3:
                    continue
                ids = ids.unsqueeze(0).cuda()
                mask = mask.unsqueeze(0).cuda()
                a_t = torch.tensor(a, dtype=torch.float32).cuda()
                out = model(input_ids=ids)
                lp = seq_logp(out.logits, ids, mask)
                with torch.no_grad(), model.disable_adapter():
                    ref_out = model(input_ids=ids)
                    ref_lp = seq_logp(ref_out.logits, ids, mask).detach()
                kl = (torch.exp(ref_lp - lp) - (ref_lp - lp) - 1.0)
                loss = -(a_t * lp) + args.kl_beta * kl
            (loss / ACC).backward()
            tot_loss += float(loss.detach())
            n_acc += 1
            if n_acc % ACC == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if n_acc % 64 == 0:
                print(f"  step {n_acc}/{len(traj)} loss~{tot_loss/n_acc:.4f}", flush=True)
        if n_acc % ACC != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        stat["train_wall_s"] = round(time.perf_counter() - t_train, 1)
        stat["n_turns_no_span"] = n_turns_no_span
        if args.adv == "gigpo" and mag_den:
            stat["mean_abs_A_tok"] = round(mag_num / mag_den, 4)

        # ---- 4. 保存 + 合并 ----
        lora_dir = f"{out_dir}/iter{it}{tag}_lora"
        merged_dir = f"{out_dir}/iter{it}{tag}_merged"
        model.save_pretrained(lora_dir)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        del model, base, merged, opt
        gc.collect()
        torch.cuda.empty_cache()

        stat["loss"] = round(tot_loss / max(n_acc, 1), 4)
        stat["iter_wall_s"] = round(time.perf_counter() - t_iter, 1)
        with open(f"{out_dir}/stats{tag}.jsonl", "a") as f:
            f.write(json.dumps(stat, ensure_ascii=False) + "\n")
        print(json.dumps({"saved": merged_dir}, ensure_ascii=False), flush=True)

    print(f"C4 {args.adv.upper()} training done.")


if __name__ == "__main__":
    main()
