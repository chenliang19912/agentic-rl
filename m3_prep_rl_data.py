"""M3 RL 训练数据准备：从 HotpotQA train 采样，与 M2 SFT 数据严格不重叠

输出 data/rl_train.jsonl：{"question","answer","gold_titles":[...]}
"""
import json

import pandas as pd

SFT_DATA = "data/sft_train.jsonl"
OUT = "data/rl_train.jsonl"
N = 4000
SEED = 23  # 与 M2 的 seed=7 不同，且下面显式剔除重叠问题


def main():
    sft_qs = {json.loads(l)["meta"]["q"] for l in open(SFT_DATA)}
    dfs = [pd.read_parquet(p) for p in
           ["data/hotpot_train_0.parquet", "data/hotpot_train_1.parquet"]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["type"] == "bridge"].reset_index(drop=True)
    df = df[~df["question"].isin(sft_qs)]
    rows = df.sample(n=min(N, len(df)), random_state=SEED)
    with open(OUT, "w") as f:
        for _, r in rows.iterrows():
            f.write(json.dumps({
                "question": r["question"],
                "answer": r["answer"],
                "gold_titles": list(r["supporting_facts"]["title"]),
            }, ensure_ascii=False) + "\n")
    print(f"saved {len(rows)} RL prompts to {OUT} (excluded {len(sft_qs)} SFT questions)")


if __name__ == "__main__":
    main()
