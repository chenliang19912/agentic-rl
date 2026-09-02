"""M2 SFT 数据 v2：证据落地版（修复 v1 的两个根因）

v1 问题：80% 轨迹的检索结果不含 gold 段落，却仍教模型直接作答
→ 模型学到"无视证据"，且没有"换 query 重搜"的示范。

v2 策略（保证每条轨迹答案都有证据支撑）：
- q1 = question；之后若仍有 gold 段落未覆盖，
  用"缺失段落 title + question"反构 query 再搜（BM25 对含 title 词的
  查询几乎必中），最多 3 次搜索
- q1 即覆盖全部 gold → 单跳轨迹；需要重搜 → 多跳轨迹（教"精炼再搜"行为）
- 3 次仍未覆盖则丢弃（极少）

输出 data/sft_train_v2.jsonl
"""
import json

import pandas as pd

from m1_baseline_c import BM25, normalize_answer

TRAIN = ["data/hotpot_train_0.parquet", "data/hotpot_train_1.parquet"]
OUT = "data/sft_train_v2.jsonl"
TOP_K = 6
TARGET = 5000
SEED = 11
MAX_SEARCH = 3

SYSTEM = (
    "You are a question-answering agent. Answer the question using search results.\n"
    "To search, output exactly: <search>your query</search>\n"
    "When you can answer, output exactly: <answer>your answer</answer>\n"
    "Search at most 3 times. Answer must be concise."
)


def fmt_results(query, paras):
    lines = "\n".join(f"- {t}: {x}" for t, x in paras) or "(no results)"
    return f"Search results for '{query}':\n{lines}"


def main():
    # 排除 v1 SFT 与 RL 用题，保持三个集合互斥
    exclude = {json.loads(l)["meta"]["q"] for l in open("data/sft_train.jsonl")}
    exclude |= {json.loads(l)["question"] for l in open("data/rl_train.jsonl")}

    dfs = [pd.read_parquet(p) for p in TRAIN]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["type"] == "bridge"].reset_index(drop=True)
    df = df[~df["question"].isin(exclude)]
    df = df.sample(n=min(9000, len(df)), random_state=SEED).reset_index(drop=True)
    print(f"candidate pool: {len(df)}")

    bm25 = BM25()
    kept, stats = [], {"one_hop": 0, "multi_hop": 0, "drop": 0}

    for i, row in df.iterrows():
        if len(kept) >= TARGET:
            break
        q, a = row["question"], row["answer"]
        gold_titles_raw = list(row["supporting_facts"]["title"])
        gold_norm = {normalize_answer(t) for t in gold_titles_raw}

        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {q}"},
        ]
        found = set()
        queries = []
        for step in range(MAX_SEARCH):
            if step == 0:
                qi = q
            else:
                missing = [t for t in gold_titles_raw
                           if normalize_answer(t) not in found]
                qi = f"{missing[0]} {q}"
            paras = [bm25.get_doc(d) for d in bm25.search(qi, k=TOP_K)]
            found |= {normalize_answer(t) for t, _ in paras}
            queries.append((qi, paras))
            if gold_norm <= found:
                break
        if not (gold_norm <= found):
            stats["drop"] += 1
            continue
        for qi, paras in queries:
            msgs.append({"role": "assistant", "content": f"<search>{qi}</search>"})
            msgs.append({"role": "user", "content": fmt_results(qi, paras)})
        msgs.append({"role": "assistant", "content": f"<answer>{a}</answer>"})
        kind = "one_hop" if len(queries) == 1 else "multi_hop"
        stats[kind] += 1
        kept.append({"messages": msgs,
                     "meta": {"q": q, "answer": a, "type": kind,
                              "n_search": len(queries)}})
        if len(kept) % 500 == 0:
            print(f"kept {len(kept)} ...", flush=True)

    with open(OUT, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"kept": len(kept), **stats}, indent=1))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
