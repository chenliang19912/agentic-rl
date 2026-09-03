"""中文迁移验证子实验 · SFT 数据 v1style（教师轨迹式 / 允许无证据作答）

背景（详见 M0_README 踩坑⑥）：证据落地版三连败（截断→4096→rebalance+5ep），
作答率始终 0%。诊断确认：100% 落地证据的数据教会模型「见到 gold 证据才作答」
的条件策略，而真实检索 top-10 全中两个 gold 标题仅 16.3% → 条件几乎永不触发
→ 搜索循环。这与英文线 v2（证据落地版）的失败模式和根因完全一致。

英文线验证过的对策：**回退教师轨迹式作为 RL 起点**——答案无条件跟在搜索后
（即使检索结果没有 gold），让「搜索→作答」的终止决策成为无条件先验，
把「何时作答」的质量问题交给 RL 奖励塑形（英文 v1 温度采样 ~20% 作答率，
足够当 GRPO 火种）。

本脚本复刻英文 v1 配方（5000 条、一半单跳一半两跳、1 epoch），按中文语料情况：
- 源：data/zh_anchored_train.jsonl（8,578 条锚定训练题，与 dev 天然不重叠）
- 排除现有 RL 池问题（data/rl_train_zh.jsonl 的 518 题），保持四集互斥
- one_hop：search[问题] → 真实检索结果 → 直接答 gold（大多数无证据，正是目的）
- two_hop：search[问题] → search[第二个规范标题（短实体 query）] → 答 gold
- 共 4000 条：2000 one_hop + 2000 two_hop（one_hop 占比 50% 给足终止信号）

搜索结果包装文本与 SearchEnv(lang="zh") 逐字符一致（训练/推理同分布）。
"""
import json
import random

from m1_baseline_c import BM25
from m3_env import SYSTEM_ZH

SRC = "data/zh_anchored_train.jsonl"
RL_POOL = "data/rl_train_zh.jsonl"
IDX = "index_zh"
OUT = "data/sft_train_zh_v1style.jsonl"
TOP_K = 6
N_ONE = 2000
N_TWO = 2000
SEED = 7


def fmt_results(query, paras):
    lines = "\n".join(f"- {t}: {x}" for t, x in paras) or "(无结果)"
    return f"'{query}' 的搜索结果：\n{lines}"


def main():
    rows = [json.loads(l) for l in open(SRC)]
    rl_qs = {json.loads(l)["question"] for l in open(RL_POOL)}
    rows = [r for r in rows if r["question_zh"] not in rl_qs]
    print(f"anchored candidates (excl. RL pool): {len(rows)}")

    rng = random.Random(SEED)
    rng.shuffle(rows)

    bm25 = BM25(IDX)
    out, stats = [], {"one_hop": 0, "two_hop": 0}

    for r in rows:
        if stats["one_hop"] >= N_ONE and stats["two_hop"] >= N_TWO:
            break
        q, a = r["question_zh"], r["answer_zh"]
        canon = r["titles_zh_canon"]
        want_two = stats["two_hop"] < N_TWO and len(canon) >= 2

        msgs = [
            {"role": "system", "content": SYSTEM_ZH},
            {"role": "user", "content": f"Question: {q}"},
        ]
        # 第一跳：问题原文（真实检索，不保证命中——这正是「无证据作答」示范）
        paras = [bm25.get_doc(d) for d in bm25.search(q, k=TOP_K)]
        msgs.append({"role": "assistant", "content": f"<search>{q}</search>"})
        msgs.append({"role": "user", "content": fmt_results(q, paras)})

        if want_two:
            # 第二跳：短实体 query（教「换个短问法再搜」；答案仍无条件跟上）
            q2 = canon[-1]
            paras2 = [bm25.get_doc(d) for d in bm25.search(q2, k=TOP_K)]
            msgs.append({"role": "assistant", "content": f"<search>{q2}</search>"})
            msgs.append({"role": "user", "content": fmt_results(q2, paras2)})

        msgs.append({"role": "assistant", "content": f"<answer>{a}</answer>"})
        kind = "two_hop" if want_two else "one_hop"
        stats[kind] += 1
        out.append({"messages": msgs,
                    "meta": {"q": q, "answer": a, "type": kind,
                             "n_search": 2 if want_two else 1}})

    rng.shuffle(out)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"total": len(out), **stats}, indent=1))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
