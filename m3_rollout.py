"""M3 批量多轮搜索 rollout 引擎（vLLM + SearchEnv）

对每个问题采样 G 条多轮轨迹（GRPO 组），同步锁步推进：
每轮对仍活跃的序列生成 → 解析 <search>/<answer> → 检索 → 续生成，
直到 </answer> 或达到 max_turns。

输出轨迹含：完整 messages（供 HF 训练模板化）、各轮原始文本、检索 title、
以及 compute_reward 所需的全部字段。

用法（评测/M3 共用）：
docker run --rm --gpus '"device=0"' ... python m3_rollout.py \
  --model checkpoints/m2_sft_merged --data eval --n 200 --G 1
"""
import argparse
import json
import random

import pandas as pd
from vllm import LLM, SamplingParams

from m3_env import SYSTEM, SYSTEM_ZH, SearchEnv, compute_reward, parse_trajectory

MAX_TURNS = 4      # 最多 4 个 assistant 轮（3 搜 + 1 答）
MAX_SEARCH = 3     # 搜索次数硬上限（与 SYSTEM 提示、奖励惩罚一致）
MAX_NEW_TOKENS = 200


def load_eval_questions(data="eval", n=200, seed=42):
    """评测题：HotpotQA dev bridge（与基线同 seed）；也支持 2wiki/bamboogle"""
    if data == "eval":
        df = pd.read_parquet("data/hotpot_dev_distractor.parquet")
        df = df[df["type"] == "bridge"]
        rows = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
        out = []
        for _, r in rows.iterrows():
            out.append({
                "question": r["question"], "answer": r["answer"],
                "gold_titles": set(t.lower() for t in r["supporting_facts"]["title"]),
                "source": "hotpot_dev",
            })
        return out
    if data == "bamboogle":
        out = []
        for line in open("data/bamboogle_test.jsonl"):
            o = json.loads(line)
            out.append({"question": o["question"],
                        "answer": o["golden_answers"][0],
                        "gold_answers": o["golden_answers"],
                        "gold_titles": set(), "source": "bamboogle"})
        return out[:n]
    if data == "2wiki":
        out = []
        for line in open("data/2wiki_dev.jsonl"):
            o = json.loads(line)
            out.append({"question": o["question"], "answer": o["answer"],
                        "gold_titles": set(), "source": "2wiki_dev"})
        return out[:n]
    if data == "eval_zh":
        # 中文迁移验证评测集：与英文主评测同题（dev bridge, seed=42 的翻译），
        # 且仅保留两个标题均锚定到中文维基的样本；gold 用规范标题与语料对齐
        out = []
        for line in open("data/hotpot_zh_dev.jsonl"):
            o = json.loads(line)
            out.append({"question": o["question_zh"], "answer": o["answer_zh"],
                        "gold_titles": set(t.lower() for t in
                                           o.get("titles_zh_canon") or o["titles_zh"]),
                        "source": "hotpot_zh_dev"})
        return out[:n]
    raise ValueError(data)


def rollout(model_path, questions, G=1, top_k=6, max_seqs=256,
            temperature=1.0, dtype="float16", gpu_util=0.7, max_model_len=8192,
            idx_dir="index", system=None, lang="en",
            max_new_tokens=MAX_NEW_TOKENS):
    """返回每个问题 G 条轨迹（含奖励）。
    idx_dir：检索索引目录（中文实验用 index_zh）；lang：环境回复文本语言
    max_new_tokens：每轮生成上限。英文线恒为 200（已定稿数字不可动）；
    中文线 v5 起模型带 <think> 推理，200 会被思考块吃光，用 512"""
    llm = LLM(model=model_path, dtype=dtype, gpu_memory_utilization=gpu_util,
              max_model_len=max_model_len, enforce_eager=True)
    tok = llm.get_tokenizer()
    env = SearchEnv(top_k=top_k, idx_dir=idx_dir, lang=lang)
    sys_prompt = system or SYSTEM
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                        top_p=1.0, stop=["</search>", "</answer>"])

    # 展开为 N×G 个序列
    seqs = []
    for qi, item in enumerate(questions):
        for _ in range(G):
            seqs.append({
                "qi": qi,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Question: {item['question']}"},
                ],
                "turn_texts": [], "retrieved_titles": [], "done": False,
            })
    active = list(range(len(seqs)))

    for turn in range(MAX_TURNS):
        if not active:
            break
        prompts = [
            tok.apply_chat_template(seqs[i]["messages"], tokenize=False,
                                    add_generation_prompt=True)
            for i in active
        ]
        # 分批防 OOM
        outs = []
        for b0 in range(0, len(prompts), max_seqs):
            outs += llm.generate(prompts[b0:b0 + max_seqs], sp)
        next_active = []
        for j, i in enumerate(active):
            raw = outs[j].outputs[0].text.strip()
            s = seqs[i]
            s["turn_texts"].append(raw)
            parsed = parse_trajectory([raw])[0]
            if parsed[0] == "search" and len(s["retrieved_titles"]) < MAX_SEARCH:
                query = parsed[1] or "search"
                result_text, titles = env.search(query)
                s["retrieved_titles"].append(titles)
                s["messages"].append({"role": "assistant",
                                      "content": f"<search>{query}</search>"})
                s["messages"].append({"role": "user", "content": result_text})
                next_active.append(i)
            else:
                # answer / invalid / 搜索超限：收尾，交给格式奖励惩罚。
                # 规范化存储：补上被 vLLM stop 剥掉的闭合标签
                kind, content, _ = parsed
                if kind == "search":
                    stored = f"<search>{content or 'search'}</search>"
                elif kind == "answer":
                    stored = f"<answer>{content}</answer>"
                else:
                    stored = raw
                s["messages"].append({"role": "assistant", "content": stored})
                s["done"] = True
        active = next_active
    # 达到 max_turns 仍未收尾的序列：原样结束（格式分 0）

    # 计算奖励
    results = []
    for s in seqs:
        item = questions[s["qi"]]
        answers = item.get("gold_answers") or [item["answer"]]
        reward, info = -1e9, None
        for a in answers:   # 多参考答案取最优（bamboogle 等）
            r, inf = compute_reward(
                s["turn_texts"], a,
                item.get("gold_titles", set()),
                s["retrieved_titles"], max_search=3, max_turns=MAX_TURNS,
            )
            if r > reward:
                reward, info = r, inf
        results.append({
            "qi": s["qi"], "question": item["question"],
            "gold": item["answer"], "messages": s["messages"],
            "reward": reward, **info,
        })
    return results, llm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/m2_sft_merged")
    ap.add_argument("--data", default="eval")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--G", type=int, default=1)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--idx_dir", default="index")
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    questions = load_eval_questions(args.data, args.n)
    zh = args.data == "eval_zh"
    results, _ = rollout(args.model, questions, G=args.G, top_k=args.top_k,
                         temperature=args.temperature, idx_dir=args.idx_dir,
                         system=SYSTEM_ZH if zh else SYSTEM,
                         lang="zh" if zh else "en",
                         max_new_tokens=args.max_new_tokens)
    n = len(questions)
    em_like = sum(1 for r in results if r["r_answer"] == 1.0) / len(results)
    avg_r = sum(r["reward"] for r in results) / len(results)
    avg_f1 = sum(r["r_answer"] for r in results) / len(results)
    avg_search = sum(r["n_search"] for r in results) / len(results)
    fmt = sum(r["r_format"] for r in results) / len(results)
    print(json.dumps({"model": args.model, "data": args.data, "n_q": n,
                      "avg_reward": round(avg_r, 4), "avg_answer_f1": round(avg_f1, 4),
                      "pct_perfect_answer": round(em_like * 100, 2),
                      "format_ok_pct": round(fmt * 100, 2),
                      "avg_n_search": round(avg_search, 2)}, ensure_ascii=False, indent=1))
    out = args.out or f"logs/rollout_{args.data}_{n}.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"saved {len(results)} trajectories to {out}")
