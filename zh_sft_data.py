"""中文迁移验证子实验 · 第三步：证据落地中文 SFT 数据 + RL 提示池

与英文管线 m2_sft_data_v2.py 完全同构，仅语言/索引/数据源不同：
- 输入：data/zh_anchored_train.jsonl（两个翻译标题均锚定到中文维基的样本）
- q1 = question_zh；未覆盖 gold 时用"缺失标题 + 问题"反构 query 再搜
  （BM25 对含标题词的查询几乎必中），最多 3 次搜索
- 3 次仍未覆盖则丢弃
- 前 TARGET_SFT 条 → data/sft_train_zh.jsonl（SFT 冷启动）
- 其余 → data/rl_train_zh.jsonl（RL 提示池，与 SFT 严格不重叠）

搜索结果的包装文本与 SearchEnv(lang="zh") 逐字符一致（训练/推理同分布）。
"""
import json

from m1_baseline_c import BM25, normalize_answer
from m3_env import SYSTEM_ZH
from zh_anchor import char_bigrams

SRC = "data/zh_anchored_train.jsonl"
IDX = "index_zh"
OUT_SFT = "data/sft_train_zh.jsonl"
OUT_RL = "data/rl_train_zh.jsonl"
TOP_K = 6
TARGET_SFT = 2000
MAX_SEARCH = 3


def fmt_results(query, paras):
    lines = "\n".join(f"- {t}: {x}" for t, x in paras) or "(无结果)"
    return f"'{query}' 的搜索结果：\n{lines}"


def answer_supported(answer, queries):
    """答案证据核验：答案的字 bigram 须能在检索到的段落中找到足够覆盖。
    中文译名/译法常有变体，标题锚定只能保证文章对，不能保证答案字面存在；
    覆盖不足说明该问题在当前语料下不可答，丢弃以免教出无证据作答。"""
    a = normalize_answer(answer)
    pool = "".join(x for _, paras in queries for _, x in paras)
    ab = char_bigrams(a)
    if not ab or len(a) == 1:
        return a in pool
    pb = char_bigrams(pool)
    return len(ab & pb) / len(ab) >= 0.6


def main():
    rows = [json.loads(l) for l in open(SRC)]
    print(f"anchored candidates: {len(rows)}")

    bm25 = BM25(IDX)
    pool = []   # 全部幸存者，最后再切分，避免 SFT 吃光样本
    stats = {"one_hop": 0, "multi_hop": 0, "drop": 0}

    for r in rows:
        q, a = r["question_zh"], r["answer_zh"]
        gold_titles_raw = r["titles_zh_canon"]   # 语料真实标题，覆盖检查口径一致
        gold_norm = {normalize_answer(t) for t in gold_titles_raw}

        msgs = [
            {"role": "system", "content": SYSTEM_ZH},
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
        if not answer_supported(a, queries):
            stats["no_answer_evidence"] = stats.get("no_answer_evidence", 0) + 1
            continue

        for qi, paras in queries:
            msgs.append({"role": "assistant", "content": f"<search>{qi}</search>"})
            msgs.append({"role": "user", "content": fmt_results(qi, paras)})
        msgs.append({"role": "assistant", "content": f"<answer>{a}</answer>"})
        kind = "one_hop" if len(queries) == 1 else "multi_hop"
        stats[kind] += 1
        pool.append({"messages": msgs,
                     "question": q, "answer": a,
                     "gold_titles": gold_titles_raw,
                     "meta": {"q": q, "answer": a, "type": kind,
                              "n_search": len(queries)}})

    # ---- 幸存者池切分：SFT 至多 2000 且 ≤60%，RL 至少留 192 ----
    n_sft = min(TARGET_SFT, int(len(pool) * 0.6))
    if len(pool) - n_sft < 192:
        n_sft = max(0, len(pool) - 192)
    sft_kept, rl_pool = pool[:n_sft], pool[n_sft:]

    with open(OUT_SFT, "w") as f:
        for r in sft_kept:
            f.write(json.dumps({"messages": r["messages"], "meta": r["meta"]},
                               ensure_ascii=False) + "\n")
    with open(OUT_RL, "w") as f:
        for r in rl_pool:
            f.write(json.dumps({"question": r["question"], "answer": r["answer"],
                                "gold_titles": r["gold_titles"]},
                               ensure_ascii=False) + "\n")
    print(json.dumps({"survivors": len(pool), "sft": len(sft_kept),
                      "rl_pool": len(rl_pool), **stats}, indent=1))
    print(f"saved {OUT_SFT} / {OUT_RL}")


if __name__ == "__main__":
    main()
