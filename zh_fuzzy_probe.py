"""中文锚定快速探查：① train bridge 题池总量；② dev 已翻译题的模糊锚定潜力

模糊锚定规则（在精确匹配失败后尝试，均为廉价操作）：
- 去括号消歧后缀（"X (Y)" → "X"）再精确匹配
- 标题被译名包含：枚举译名的所有 >=3 字子串，查标题集合
"""
import json
import re

import pandas as pd

from m1_baseline_c import normalize_answer

RAW = "data/zhwiki_paras.jsonl"
DEV = "data/zh_translated_dev.jsonl"

_PAREN_RE = re.compile(r"[（(][^（()）]*[）)]$")


def strip_paren(t):
    prev = None
    while prev != t:
        prev = t
        t = _PAREN_RE.sub("", t).strip()
    return t


def anchored(q, title_set):
    if q in title_set:
        return True
    if strip_paren(q) in title_set:
        return True
    # 标题 ⊆ 译名（枚举子串，O(L^2) 次集合查询，L<=20 很廉价）
    L = len(q)
    for i in range(L - 2):
        for j in range(i + 3, min(i + 31, L + 1)):
            if q[i:j] in title_set:
                return True
    return False


def main():
    dfs = [pd.read_parquet(p) for p in
           ["data/hotpot_train_0.parquet", "data/hotpot_train_1.parquet"]]
    df = pd.concat(dfs, ignore_index=True)
    bridge = df[df["type"] == "bridge"]
    n2 = sum(1 for sf in bridge["supporting_facts"] if len(sf["title"]) == 2)
    print(f"train rows total={len(df):,}  bridge={len(bridge):,}  "
          f"bridge_with_2_titles={n2:,}")

    titles = set()
    for line in open(RAW):
        titles.add(normalize_answer(json.loads(line)["title"]))
    print(f"title set: {len(titles):,}")

    rows = [json.loads(l) for l in open(DEV)]
    rows = [r for r in rows if r["parse_ok"]]
    both = one = 0
    for r in rows:
        hits = [anchored(normalize_answer(t), titles) for t in r["titles_zh"]]
        both += int(hits[0] and hits[1])
        one += int(hits[0] or hits[1])
    n = len(rows)
    print(f"dev parse_ok={n}  both_anchored={both} ({both/max(n,1)*100:.1f}%)  "
          f"at_least_one={one} ({one/max(n,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
