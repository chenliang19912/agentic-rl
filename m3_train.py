"""M3 自研多轮 GRPO 训练循环（V100 sm_70 兼容，不依赖 verl）

每个 iteration：
  1. vLLM 批量多轮 rollout（每题 G 条轨迹，温度采样）
  2. 奖励 + 组内归一化优势（GRPO）
  3. 释放 vLLM → HF 模型（base + LoRA）：
     loss = -adv·mean_token_logp(assistant) + β·KL(policy‖base)
     reference = 同一 base 关闭 LoRA（PEFT disable_adapter），无需第二份权重
  4. 保存 LoRA 并合并为 merged 目录 → 下一轮 rollout 用

用法：
docker run --rm --gpus '"device=2"' \
  -v /mnt/nas3/shared/model:/models:ro \
  -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
  -e HF_HUB_OFFLINE=1 \
  1cat-vllm:v100-1.3.0 python -u m3_train.py
"""
import argparse
import gc
import json
import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3_env import SYSTEM_ZH, grpo_advantages
from m3_rollout import rollout
from mask_utils import encode_chat_assistant_only

BASE = "checkpoints/m3_run2/iter0_merged"  # run2 最优 checkpoint（贪心 EM 22.6）
OUT_DIR = "checkpoints/m3"
RL_DATA = "data/rl_train.jsonl"

# ---- 超参 ----
# run1 教训：lr 2e-5 + β 0.004 单轮即行为坍缩（作答灭绝）
# run2 教训：lr 1e-5 在成功率 86% 的组对比上仍过冲（20→86→14 三拍振荡）
# run3：更温和——成功率越高更新越要小；步数砍半；β 再翻倍锚住 iter0
N_ITERS = 4
PROMPTS_PER_ITER = 128
GROUP_SIZE = 4
LR = 5e-6
KL_BETA = 0.05
MAX_LEN = 4096   # rollout 上下文上限 8192，训练侧 4096 截断极少数超长轨迹尾部
PER_DEV_BS = 2
ACC = 8
MAX_EFFECTIVE = 192   # 每轮实际用于更新的轨迹上限（~24 个优化步，防过冲）
TEMP = 1.0
ANSWER_METRIC = "f1"   # 收敛后可切 "em"

LORA = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                  target_modules="all-linear", task_type="CAUSAL_LM")


def parse_args():
    """默认值 = run3 英文配置（行为与无参数时完全一致）；
    中文迁移实验通过 --idx-dir index_zh --system-lang zh 等复用同一训练循环"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--rl-data", default=RL_DATA)
    ap.add_argument("--idx-dir", default="index")
    ap.add_argument("--n-iters", type=int, default=N_ITERS)
    ap.add_argument("--prompts-per-iter", type=int, default=PROMPTS_PER_ITER)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--system-lang", choices=["en", "zh"], default="en")
    ap.add_argument("--gpu-util", type=float, default=0.7,
                    help="rollout 侧 vLLM gpu_memory_utilization（显存紧张时调低）")
    ap.add_argument("--max-new-tokens", type=int, default=200,
                    help="rollout 每轮生成上限（中文线带 think 推理用 512）")
    return ap.parse_args()


def load_rl_prompts(path=RL_DATA):
    return [json.loads(l) for l in open(path)]


def encode_trajectory(tok, messages):
    """返回 (input_ids, assistant_mask)，仅 assistant 内容计损失（mask_utils 已实测）"""
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
    system = SYSTEM_ZH if args.system_lang == "zh" else None
    os.makedirs(out_dir, exist_ok=True)
    prompts_pool = load_rl_prompts(args.rl_data)
    tok = AutoTokenizer.from_pretrained(args.base)
    merged_dir = args.base
    prev_fmt = 0.0

    for it in range(args.n_iters):
        print(f"\n===== ITER {it} =====", flush=True)
        batch_qs = prompts_pool[it * args.prompts_per_iter:
                                (it + 1) * args.prompts_per_iter]
        if len(batch_qs) < args.group_size * 4:
            print("not enough prompts left; stop")
            break

        # ---- 1. rollout（vLLM）----
        results, llm = rollout(merged_dir, batch_qs, G=args.group_size,
                               top_k=6, temperature=TEMP,
                               idx_dir=args.idx_dir, system=system,
                               lang=args.system_lang, gpu_util=args.gpu_util,
                               max_new_tokens=args.max_new_tokens)
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        rewards = [r["reward"] for r in results]
        adv = grpo_advantages(rewards, args.group_size)
        stat = {
            "iter": it,
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "avg_f1": round(sum(r["r_answer"] for r in results) / len(results), 4),
            "fmt_pct": round(sum(r["r_format"] for r in results) / len(results) * 100, 1),
            "avg_search": round(sum(r["n_search"] for r in results) / len(results), 2),
        }
        print(json.dumps(stat, ensure_ascii=False), flush=True)

        # 逐轮落盘 rollout（run1 崩溃后无法复盘的教训）
        with open(f"{out_dir}/iter{it}_rollouts.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 坍缩熔断：作答率较上轮腰斩即停（保住最后一个正常 checkpoint）
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
        opt = torch.optim.AdamW(model.parameters(), lr=LR)

        traj = list(zip(results, adv.tolist()))
        # 零优势轨迹不产生策略梯度（全不答组在门控过程奖励后 reward 全 0 → adv 0），
        # 跳过可省下大量前向/反向，只保留 β·KL 也无意义
        n_before = len(traj)
        traj = [(r, a) for r, a in traj if abs(a) > 1e-4]
        print(f"  effective trajectories: {len(traj)}/{n_before}", flush=True)
        if not traj:
            print("no non-zero advantage; skip update this iter", flush=True)
            stat["loss"] = 0.0
            with open(f"{out_dir}/stats.jsonl", "a") as f:
                f.write(json.dumps(stat, ensure_ascii=False) + "\n")
            continue
        # 轨迹内顺序打乱（保留组结构无所谓，优势已定），截断到步数预算
        import random
        random.shuffle(traj)
        traj = traj[:MAX_EFFECTIVE]

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

    print("GRPO training done.")


if __name__ == "__main__":
    main()
