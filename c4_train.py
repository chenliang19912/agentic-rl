"""C4 M2 主实验：多轮 GRPO 训练循环（m3_train 的姊妹篇——训练层零改动复用）

与 m3_train.py 的差异仅在环境接口与统计语义：
  - rollout: m3_rollout(检索) → c4_rollout(代码沙盒)，多带一个 hidden 判分表
  - 优势/KL/熔断/落盘纪律：原样复用（grpo_advantages 环境无关；
    disable_adapter 零显存参考 KL；零优势过滤；fmt_pct 腰斩熔断；逐轮落盘）
  - 统计: r_answer/n_search → r_pass/n_runs/submit/dup（reward hacking 监控：
    dup_ref 率逐 iter 入日志，警惕"改写绕过查重"策略演化，README §3.4）

起点：checkpoints/c4_rft_merged（RFT 冷启动，目标线=多轮贪心 0.7295）
数据：data/c4/tasks_train.jsonl（89 题，背题筛选后），每 iter 全量 × G 条

用法（docker 单卡，长任务 detached）：
  python -u c4_train.py --base checkpoints/c4_rft_merged --n-iters 4 --group-size 4
"""
import argparse
import gc
import json
import os
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3_env import grpo_advantages            # 环境无关，直接复用（检索 Agent 项目栈）
from c4_rollout import rollout, load_tasks
from mask_utils import encode_chat_assistant_only

# ---- 超参（检索 Agent 项目 run3 教训直接沿用：成功率越高更新越要小）----
N_ITERS = 4
GROUP_SIZE = 4
LR = 5e-6
KL_BETA = 0.05
MAX_LEN = 4096
ACC = 8                # 梯度累积（有效 batch 8）
MAX_EFFECTIVE = 192    # 每 iter 实际更新的轨迹上限（防过冲，~24 个优化步）
TEMP = 1.0

LORA = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                  target_modules="all-linear", task_type="CAUSAL_LM")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/c4_rft_merged")
    ap.add_argument("--out", default="checkpoints/c4_grpo")
    ap.add_argument("--tasks", default="data/c4/tasks_train.jsonl")
    ap.add_argument("--hidden", default="data/c4/hidden_train.jsonl")
    ap.add_argument("--n-iters", type=int, default=N_ITERS)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--kl-beta", type=float, default=KL_BETA)
    ap.add_argument("--max-effective", type=int, default=MAX_EFFECTIVE)
    ap.add_argument("--gpu-util", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    return ap.parse_args()


def encode_trajectory(tok, messages):
    """(input_ids, assistant_mask)，仅 assistant 内容计损失（mask_utils 已实测）"""
    ids, _, labels = encode_chat_assistant_only(tok, messages, max_len=MAX_LEN)
    mask = [1 if l != -100 else 0 for l in labels]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def seq_logp(logits, input_ids, mask):
    """assistant token 平均 logp（长度归一，避免长轨迹主导）"""
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tok_logp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:]
    return (tok_logp * m).sum(-1) / m.sum(-1).clamp(min=1)


def main():
    args = parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    tasks, hidden = load_tasks(args.tasks, args.hidden)
    print(f"train tasks: {len(tasks)} | base: {args.base}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    merged_dir = args.base
    prev_fmt = 0.0

    for it in range(args.n_iters):
        print(f"\n===== ITER {it} =====", flush=True)
        # 89 题小池：每 iter 全量（题多时改为按 seed=it 随机抽子集）
        batch = tasks

        # ---- 1. rollout（vLLM 锁步多轮 + 沙盒 + grading）----
        results, llm = rollout(merged_dir, batch, hidden, G=args.group_size,
                               temperature=TEMP, gpu_util=args.gpu_util,
                               max_new_tokens=args.max_new_tokens)
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        rewards = [r["reward"] for r in results]
        adv = grpo_advantages(rewards, args.group_size)
        stat = {
            "iter": it,
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "avg_pass": round(sum(r["r_pass"] for r in results) / len(results), 4),
            "fmt_pct": round(sum(r["r_format"] for r in results) / len(results) * 100, 1),
            "submit_pct": round(sum(1 for r in results if r["answer_code"]) / len(results) * 100, 1),
            "avg_runs": round(sum(r["n_runs"] for r in results) / len(results), 2),
            "dup_cnt": sum(1 for r in results if (r["grade"] or {}).get("dup_ref")),
            "grade_timeout_cnt": sum(1 for r in results if (r["grade"] or {}).get("timed_out")),
        }
        print(json.dumps(stat, ensure_ascii=False), flush=True)

        # 逐轮落盘 rollout（run1 崩溃后无法复盘的教训）
        with open(f"{out_dir}/iter{it}_rollouts.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 坍缩熔断：格式合法率较上轮腰斩即停（保住最后一个正常 checkpoint）
        if it > 0 and stat["fmt_pct"] < 0.5 * prev_fmt:
            print(f"ABORT: fmt_pct collapsed ({prev_fmt} -> {stat['fmt_pct']}); "
                  f"keeping last checkpoint", flush=True)
            return
        prev_fmt = stat["fmt_pct"]

        # ---- 2. HF + LoRA 策略更新 ----
        base = AutoModelForCausalLM.from_pretrained(
            merged_dir, torch_dtype=torch.float16).cuda()
        base.config.use_cache = False
        model = get_peft_model(base, LORA)
        model.gradient_checkpointing_enable()
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        traj = list(zip(results, adv.tolist()))
        # 零优势轨迹不产生策略梯度（全组同分 → adv 0），跳过省前向/反向
        n_before = len(traj)
        traj = [(r, a) for r, a in traj if abs(a) > 1e-4]
        print(f"  effective trajectories: {len(traj)}/{n_before}", flush=True)
        if not traj:
            print("no non-zero advantage; skip update this iter", flush=True)
            stat["loss"] = 0.0
            with open(f"{out_dir}/stats.jsonl", "a") as f:
                f.write(json.dumps(stat, ensure_ascii=False) + "\n")
            continue
        random.shuffle(traj)
        traj = traj[:args.max_effective]

        opt.zero_grad()
        n_acc, tot_loss = 0, 0.0
        for step, (r, a) in enumerate(traj):
            ids, mask = encode_trajectory(tok, r["messages"])
            if mask.sum() < 3:      # 无有效 assistant token
                continue
            ids = ids.unsqueeze(0).cuda()
            mask = mask.unsqueeze(0).cuda()
            a_t = torch.tensor(a, dtype=torch.float32).cuda()

            out = model(input_ids=ids)
            lp = seq_logp(out.logits, ids, mask)
            with torch.no_grad(), model.disable_adapter():
                ref_out = model(input_ids=ids)   # adapter 关闭 → 参考策略 = 冻结 base
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

        # ---- 3. 保存 + 合并，供下轮 rollout ----
        lora_dir = f"{out_dir}/iter{it}_lora"
        merged_dir = f"{out_dir}/iter{it}_merged"
        model.save_pretrained(lora_dir)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        del model, base, merged, opt
        gc.collect()
        torch.cuda.empty_cache()

        stat["loss"] = round(tot_loss / max(n_acc, 1), 4)
        with open(f"{out_dir}/stats.jsonl", "a") as f:
            f.write(json.dumps(stat, ensure_ascii=False) + "\n")
        print(json.dumps({"saved": merged_dir}, ensure_ascii=False), flush=True)

    print("C4 GRPO training done.")


if __name__ == "__main__":
    main()
