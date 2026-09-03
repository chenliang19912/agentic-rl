"""M3 对照消融：朴素 REINFORCE vs GRPO（m3_train.py run2 配置的同条件对照）

背景：2026.02《How to Train Your Deep Research Agent?》(arXiv 2602.19526)
系统比较 Search-R1 场景下 REINFORCE/PPO/GRPO，结论是 REINFORCE 最终性能与
效率最好、GRPO 稳定性最差。我们在 run2 中观测到的 20→86→14 三拍振荡与之互证。
本脚本做同条件对照：同起点、同数据、同超参（run2 配置），仅把优势估计从
"组内归一化（GRPO）"换成"全局 batch 均值基线（REINFORCE）"。

与 m3_train.py 的差异只有两处：advantage 计算 与 输出目录。
"""
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3_rollout import rollout
from mask_utils import encode_chat_assistant_only

BASE = "checkpoints/m2_sft_merged"     # 与 run2 同起点
OUT_DIR = "checkpoints/m3_reinforce"
RL_DATA = "data/rl_train.jsonl"

# ---- 超参：与 run2 完全一致，保证对照公平 ----
N_ITERS = 3
PROMPTS_PER_ITER = 128
GROUP_SIZE = 4
LR = 1e-5
KL_BETA = 0.02
MAX_LEN = 4096
ACC = 4
TEMP = 1.0

LORA = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                  target_modules="all-linear", task_type="CAUSAL_LM")


def load_rl_prompts(path=RL_DATA):
    return [json.loads(l) for l in open(path)]


def reinforce_advantages(rewards: list):
    """REINFORCE with batch-mean baseline：A_i = r_i - mean(r)，无组结构、无 std 归一化"""
    rewards = np.asarray(rewards, dtype=np.float32)
    return rewards - rewards.mean()


def encode_trajectory(tok, messages):
    ids, _, labels = encode_chat_assistant_only(tok, messages, max_len=MAX_LEN)
    mask = [1 if l != -100 else 0 for l in labels]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def seq_logp(logits, input_ids, mask):
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tok_logp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:]
    return (tok_logp * m).sum(-1) / m.sum(-1).clamp(min=1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prompts_pool = load_rl_prompts()
    tok = AutoTokenizer.from_pretrained(BASE)
    merged_dir = BASE
    prev_fmt = 0.0

    for it in range(N_ITERS):
        print(f"\n===== ITER {it} =====", flush=True)
        batch_qs = prompts_pool[it * PROMPTS_PER_ITER:(it + 1) * PROMPTS_PER_ITER]
        if len(batch_qs) < GROUP_SIZE * 4:
            print("not enough prompts left; stop")
            break

        results, llm = rollout(merged_dir, batch_qs, G=GROUP_SIZE,
                               top_k=6, temperature=TEMP)
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        rewards = [r["reward"] for r in results]
        adv = reinforce_advantages(rewards)
        stat = {
            "iter": it,
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "avg_f1": round(sum(r["r_answer"] for r in results) / len(results), 4),
            "fmt_pct": round(sum(r["r_format"] for r in results) / len(results) * 100, 1),
            "avg_search": round(sum(r["n_search"] for r in results) / len(results), 2),
            "adv_abs_mean": round(float(np.abs(adv).mean()), 4),
        }
        print(json.dumps(stat, ensure_ascii=False), flush=True)

        with open(f"{OUT_DIR}/iter{it}_rollouts.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 同 run2 的熔断：作答率腰斩即停
        if it > 0 and stat["fmt_pct"] < 0.5 * prev_fmt:
            print(f"ABORT: fmt_pct collapsed ({prev_fmt} -> {stat['fmt_pct']}); "
                  f"keeping last checkpoint", flush=True)
            return
        prev_fmt = stat["fmt_pct"]

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
            stat["loss"] = 0.0
            with open(f"{OUT_DIR}/stats.jsonl", "a") as f:
                f.write(json.dumps(stat, ensure_ascii=False) + "\n")
            continue
        import random
        random.shuffle(traj)

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

        lora_dir = f"{OUT_DIR}/iter{it}_lora"
        merged_dir = f"{OUT_DIR}/iter{it}_merged"
        model.save_pretrained(lora_dir)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        del model, base, merged, opt
        gc.collect()
        torch.cuda.empty_cache()

        stat["loss"] = round(tot_loss / max(n_acc, 1), 4)
        with open(f"{OUT_DIR}/stats.jsonl", "a") as f:
            f.write(json.dumps(stat, ensure_ascii=False) + "\n")
        print(json.dumps({"saved": merged_dir}, ensure_ascii=False), flush=True)

    print("REINFORCE ablation done.")


if __name__ == "__main__":
    main()
