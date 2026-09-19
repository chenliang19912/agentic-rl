"""M5-A1 自适应检索 k · 多轮 GRPO 训练循环（m3_train 的协议扩展副本）

对照设计（配对）：与 run3 完全同参同数据切片同冷启动（m3_run2/iter0_merged，
lr 5e-6 / β 0.05 / 128 题×G4 / 4 iters / 温度 1.0 / max_new_tokens 200），
唯一差异 = 检索协议（可选 k 动作 + 描述动作空间的 SYSTEM）。
协议、判据与结果见 M5_ADAPTIVE_K.md。

用法（docker 内）：
  python -u m5_train.py --gpu-util 0.7
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

from m3_env import grpo_advantages
from m5_env import SYSTEM_M5
from m5_rollout import rollout
from m3_train import (ACC, KL_BETA, LORA, LR, MAX_EFFECTIVE, MAX_LEN,
                      encode_trajectory, load_rl_prompts, seq_logp)

BASE = "checkpoints/m3_run2/iter0_merged"   # 与 run3 同冷启动（配对对照）
OUT_DIR = "checkpoints/m5"
RL_DATA = "data/rl_train.jsonl"
TEMP = 1.0
N_ITERS = 4
PROMPTS_PER_ITER = 128
GROUP_SIZE = 4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--rl-data", default=RL_DATA)
    ap.add_argument("--idx-dir", default="index")
    ap.add_argument("--n-iters", type=int, default=N_ITERS)
    ap.add_argument("--prompts-per-iter", type=int, default=PROMPTS_PER_ITER)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--gpu-util", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    prompts_pool = load_rl_prompts(args.rl_data)
    tok = AutoTokenizer.from_pretrained(args.base)
    merged_dir = args.base
    prev_fmt = 0.0

    for it in range(args.n_iters):
        print(f"\n===== M5 ITER {it} =====", flush=True)
        batch_qs = prompts_pool[it * args.prompts_per_iter:
                                (it + 1) * args.prompts_per_iter]
        if len(batch_qs) < args.group_size * 4:
            print("not enough prompts left; stop")
            break

        # ---- 1. rollout（vLLM，自适应 k 环境）----
        results, llm = rollout(merged_dir, batch_qs, G=args.group_size,
                               temperature=TEMP, idx_dir=args.idx_dir,
                               system=SYSTEM_M5, gpu_util=args.gpu_util,
                               max_new_tokens=args.max_new_tokens)
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        rewards = [r["reward"] for r in results]
        adv = grpo_advantages(rewards, args.group_size)
        n_searched = sum(1 for r in results if r["n_search"])
        stat = {
            "iter": it,
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "avg_f1": round(sum(r["r_answer"] for r in results) / len(results), 4),
            "fmt_pct": round(sum(r["r_format"] for r in results) / len(results) * 100, 1),
            "avg_search": round(sum(r["n_search"] for r in results) / len(results), 2),
            "k_specified_pct": round(sum(r["k_specified_pct"] for r in results)
                                     / max(1, n_searched) * 100, 1),
            "avg_k": round(sum(r["avg_k"] for r in results if r["n_search"])
                           / max(1, n_searched), 2),
            "avg_search_chars": round(sum(r["search_chars"] for r in results)
                                      / len(results), 1),
        }
        print(json.dumps(stat, ensure_ascii=False), flush=True)

        with open(f"{out_dir}/iter{it}_rollouts.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 坍缩熔断（与 m3_train 同纪律）
        if it > 0 and stat["fmt_pct"] < 0.5 * prev_fmt:
            print(f"ABORT: fmt_pct collapsed ({prev_fmt} -> {stat['fmt_pct']}); "
                  f"keeping last checkpoint", flush=True)
            return
        prev_fmt = stat["fmt_pct"]

        # ---- 2. HF + LoRA 策略更新（与 m3_train 逐行同逻辑）----
        base = AutoModelForCausalLM.from_pretrained(
            merged_dir, torch_dtype=torch.float16).cuda()
        base.config.use_cache = False
        model = get_peft_model(base, LORA)
        model.gradient_checkpointing_enable()
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR)

        traj = list(zip(results, adv.tolist()))
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
        traj = traj[:MAX_EFFECTIVE]

        opt.zero_grad()
        n_acc, tot_loss = 0, 0.0
        for step, (r, a) in enumerate(traj):
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
            loss = -(a_t * lp) + KL_BETA * kl
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

        # ---- 3. 保存 + 合并 ----
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

    print("M5 GRPO training done.")


if __name__ == "__main__":
    main()
