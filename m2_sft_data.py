"""M2 SFT 冷启动数据生成：多轮 <search>/<answer> 轨迹

设计原则：与 M3 RL rollout 环境完全同构（同一 SYSTEM、同一检索返回格式、同一 top-k），
避免 SFT→RL 分布错位。

轨迹两类：
- A 单跳（~50%）：Q → <search>Q</search> → 环境返回 BM25 top-k → <answer>A</answer>
- B 两跳（~50%）：Q → <search>Q</search> → 环境返回 → 教师模型（Qwen3-4B-Instruct-2507）
  生成第二跳 query → 环境返回 → <answer>A</answer>
  （教师只生成 query，答案永远用 gold，不引入教师幻觉）

输出 data/sft_train.jsonl：{"messages": [...], "meta": {...}}
SFT 训练时仅对 assistant 轮计损失（trl SFTTrainer 对话格式自动 mask）。

用法（宿主机，依赖 index/ 已构建）：
docker run --rm --gpus '"device=1"' \
  -v /mnt/nas3/shared/model:/models:ro \
  -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
  -e HF_HUB_OFFLINE=1 \
  1cat-vllm:v100-1.3.0 python m2_sft_data.py --n 5000
"""
import argparse
import json
import random
import re

import pandas as pd
from vllm import LLM, SamplingParams

from m1_baseline_c import BM25, normalize_answer, tokenize  # 复用

TEACHER = "/models/Qwen3-4B-Instruct-2507"
TRAIN = ["data/hotpot_train_0.parquet", "data/hotpot_train_1.parquet"]
OUT = "data/sft_train.jsonl"
TOP_K = 6  # 与 M3 RL rollout 环境保持一致

SYSTEM = (
    "You are a question-answering agent. Answer the question using search results.\n"
    "To search, output exactly: <search>your query</search>\n"
    "When you can answer, output exactly: <answer>your answer</answer>\n"
    "Search at most 3 times. Answer must be concise."
)

TEACHER_SYS = (
    "You are helping build a search agent. Given a multi-hop question, the gold final answer, "
    "and the titles of documents returned by a first search, write ONE short follow-up search "
    "query (<= 12 words) to find the missing supporting fact. Output ONLY the query, nothing else."
)


def fmt_results(query, paras):
    lines = "\n".join(f"- {t}: {x}" for t, x in paras) or "(no results)"
    return f"Search results for '{query}':\n{lines}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--two_hop_ratio", type=float, default=0.5)
    args = ap.parse_args()
    random.seed(args.seed)

    dfs = [pd.read_parquet(p) for p in TRAIN]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["type"] == "bridge"].reset_index(drop=True)
    rows = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"sampled {len(rows)} bridge train rows (of {len(df):,})")

    bm25 = BM25()

    n_two = int(len(rows) * args.two_hop_ratio)
    two_hop_idx = set(random.sample(range(len(rows)), n_two))

    # ---- 第一跳检索（所有样本）----
    print("first-hop retrieval...")
    paras1_all = []
    for i, q in enumerate(rows["question"]):
        paras1_all.append([bm25.get_doc(d) for d in bm25.search(q, k=TOP_K)])
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(rows)}")

    # ---- 教师生成第二跳 query（两跳子集）----
    two_rows = rows.loc[sorted(two_hop_idx)]
    teacher_prompts = []
    for i, row in two_rows.iterrows():
        titles = "; ".join(t for t, _ in paras1_all[i][:4])
        teacher_prompts.append(
            f"Question: {row['question']}\nGold final answer: {row['answer']}\n"
            f"First-search document titles: {titles}\nFollow-up search query:"
        )
    print(f"generating {len(teacher_prompts)} teacher second-hop queries...")
    llm = LLM(model=TEACHER, dtype="float16", gpu_memory_utilization=0.6,
              max_model_len=2048, enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=40, temperature=0.7, top_p=0.9)
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": TEACHER_SYS}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
            extra_kwargs={"enable_thinking": False},
        )
        for p in teacher_prompts
    ]
    q2_list = [o.outputs[0].text.strip().strip('"').split("\n")[0][:120]
               for o in llm.generate(prompts, sp)]
    del llm  # 释放显存给第二跳检索无影响（检索在 CPU）

    q2_map = dict(zip(sorted(two_hop_idx), q2_list))

    # ---- 第二跳检索 ----
    print("second-hop retrieval...")
    paras2_map = {}
    for i, q2 in q2_map.items():
        paras2_map[i] = [bm25.get_doc(d) for d in bm25.search(q2, k=TOP_K)]

    # ---- 组装轨迹 ----
    gold_titles = {}
    for i, sf in enumerate(rows["supporting_facts"]):
        gold_titles[i] = {normalize_answer(t) for t in sf["title"]}

    n_gold_hit, n_format_bad = 0, 0
    with open(OUT, "w") as f:
        for i, row in rows.iterrows():
            q, a = row["question"], row["answer"]
            p1 = paras1_all[i]
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {q}"},
                {"role": "assistant", "content": f"<search>{q}</search>"},
                {"role": "user", "content": fmt_results(q, p1)},
            ]
            q2 = q2_map.get(i)
            if q2 and tokenize(q2):
                p2 = paras2_map[i]
                msgs += [
                    {"role": "assistant", "content": f"<search>{q2}</search>"},
                    {"role": "user", "content": fmt_results(q2, p2)},
                ]
                seen = {normalize_answer(t) for t, _ in p1} | {normalize_answer(t) for t, _ in p2}
            else:
                if i in two_hop_idx:   # 只有两跳样本的教师 query 失败才算 bad
                    n_format_bad += 1
                seen = {normalize_answer(t) for t, _ in p1}
            msgs.append({"role": "assistant", "content": f"<answer>{a}</answer>"})
            hit = len(seen & gold_titles[i]) / max(len(gold_titles[i]), 1)
            n_gold_hit += int(hit == 1.0)
            f.write(json.dumps(
                {"messages": msgs,
                 "meta": {"q": q, "answer": a, "type": "two_hop" if i in two_hop_idx else "one_hop",
                          "gold_para_recall": round(hit, 2)}},
                ensure_ascii=False) + "\n")

    print(json.dumps({
        "total": len(rows),
        "two_hop": len(two_hop_idx),
        "teacher_query_bad": n_format_bad,
        "gold_para_full_recall_pct": round(n_gold_hit / len(rows) * 100, 2),
    }, indent=1))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
