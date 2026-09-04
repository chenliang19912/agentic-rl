"""诊断脚本：中文 BM25（index_zh）对门控评测集 135 题的 gold 标题召回率

背景：中文 SFT v3（证据落地数据）作答率仍 0%。怀疑训练数据 100% 带落地证据，
而真实检索召回极低，模型学会"见到 gold 证据才作答"→ 真实环境永远不满足 → 搜索循环。
本脚本量化：真实检索（问题原文作 query，top-10）能召回多少 gold 标题。
对照组：用 gold 标题本身作 query 的召回（锚定正确性的冒烟测试）。
"""
import json

from m3_env import SearchEnv

env = SearchEnv(top_k=10, idx_dir="index_zh", lang="zh")
dev = [json.loads(l) for l in open("data/hotpot_zh_dev.jsonl")]
n = len(dev)

q_any = q_all = t_any = 0
for d in dev:
    gold = set(t.lower() for t in (d.get("titles_zh_canon") or d["titles_zh"]))
    # 问题原文作 query
    titles = set()
    for doc_id in env.bm25.search(d["question_zh"], k=10):
        t, _ = env.bm25.get_doc(doc_id)
        titles.add(t.strip().strip('"').lower())
    if gold & titles:
        q_any += 1
    if gold <= titles:
        q_all += 1
    # gold 标题自身作 query（冒烟：锚定标题的条目能否被检索到）
    for g in gold:
        hits = set()
        for doc_id in env.bm25.search(g, k=5):
            t, _ = env.bm25.get_doc(doc_id)
            hits.add(t.strip().strip('"').lower())
        if g in hits:
            t_any += 1

print(json.dumps({
    "n": n,
    "question_query_hit_any": round(q_any / n * 100, 1),
    "question_query_hit_all": round(q_all / n * 100, 1),
    "title_self_retrievable_pct": round(t_any / (2 * n) * 100, 1),
}, ensure_ascii=False, indent=1))
