"""中文迁移验证子实验 · 评测集过滤：只保留"语料内可答"的锚定题

检索锚定（标题变体匹配）存在误配风险；且部分问题虽标题锚定成功，
答案字面却不在语料证据中（译名变体/条目内容差异）。评测集若包含
语料内不可答的题，会系统性低估所有方法。

本脚本对 zh_anchored_dev.jsonl 逐题做与 SFT 数据构建相同的两重验证：
1. 证据覆盖：以 question_zh 及"缺失规范标题+问题"反构 query 检索（≤3 次），
   两个规范标题均须出现在检索结果中；
2. 答案证据：答案字（含 bigram 覆盖）须能在证据段落中找到。

输出 data/hotpot_zh_dev.jsonl（m3_rollout.py --data eval_zh 直接读取）。
"""
import json

from m1_baseline_c import BM25, normalize_answer
from zh_anchor import char_bigrams
from zh_sft_data import TOP_K, MAX_SEARCH

SRC = "data/zh_anchored_dev.jsonl"
OUT = "data/hotpot_zh_dev.jsonl"


def answer_supported(answer, queries):
    a = normalize_answer(answer)
    pool = "".join(x for _, paras in queries for _, x in paras)
    ab = char_bigrams(a)
    if not ab or len(a) == 1:
        return a in pool
    pb = char_bigrams(pool)
    return len(ab & pb) / len(ab) >= 0.6


def main():
    rows = [json.loads(l) for l in open(SRC)]
    print(f"anchored dev: {len(rows)}")
    bm25 = BM25("index_zh")

    kept, stats = [], {"uncovered": 0, "no_answer_evidence": 0}
    for r in rows:
        q = r["question_zh"]
        gold_titles_raw = r["titles_zh_canon"]
        gold_norm = {normalize_answer(t) for t in gold_titles_raw}
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
            stats["uncovered"] += 1
            continue
        if not answer_supported(r["answer_zh"], queries):
            stats["no_answer_evidence"] += 1
            continue
        kept.append(r)

    with open(OUT, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"kept": len(kept), **stats}))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
