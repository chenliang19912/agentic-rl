"""M3 搜索环境 + 奖励函数（与 M0 rollout / M2 SFT 数据同构）

SearchEnv：BM25 检索环境，返回格式与 m0_rollout / m2_sft_data 完全一致。
compute_reward：R = R_answer + R_format + R_process − R_over_retrieval
  - R_answer：最终答案对 gold 的 F1（软奖励起步；可用 EM 硬奖励切换）
  - R_format：所有 轮次 均以合法 <search>...</search> 或 <answer>...</answer> 结束
  - R_process：检索结果命中 gold 支撑段（title 匹配）按比例给小奖励（改进点①）
  - R_over_retrieval：检索轮数超 max_search 或检索 token 占比过高则惩罚
"""
import json
import math
import re
from collections import Counter

import numpy as np

from m1_baseline_c import BM25, normalize_answer, tokenize, f1 as ans_f1, exact_match

SYSTEM = (
    "You are a question-answering agent. Answer the question using search results.\n"
    "To search, output exactly: <search>your query</search>\n"
    "When you can answer, output exactly: <answer>your answer</answer>\n"
    "Search at most 3 times. Answer must be concise."
)

# 中文迁移验证子实验用：与 SYSTEM 语义逐条对应，仅语言不同
SYSTEM_ZH = (
    "你是一个问答智能体。请根据搜索结果回答问题。\n"
    "需要搜索时，严格输出：<search>你的查询</search>\n"
    "可以回答时，严格输出：<answer>你的答案</answer>\n"
    "最多搜索 3 次。答案必须简短。"
)

# vLLM 的 stop 字符串默认不进输出（且生成可能因 max_tokens 截断），
# 故闭合标签按可选处理
SEARCH_RE = re.compile(r"<search>(.*?)(?:</search>|$)", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S)


class SearchEnv:
    """BM25 检索环境。与 m0_rollout.py / m2_sft_data.py 的返回格式保持一致。"""

    def __init__(self, top_k=6, idx_dir="index", lang="en"):
        self.top_k = top_k
        self.lang = lang
        self.bm25 = BM25(idx_dir)

    def search(self, query: str):
        """返回 (环境回复文本, 命中段落 title 列表)。lang="zh" 用中文包装文本，
        与 SYSTEM_ZH / 中文 SFT 数据格式保持一致"""
        hits = self.bm25.search(query, k=self.top_k)
        paras = [self.bm25.get_doc(d) for d in hits]
        lines = "\n".join(f"- {t}: {x}" for t, x in paras)
        if self.lang == "zh":
            lines = lines or "(无结果)"
            return f"'{query}' 的搜索结果：\n{lines}", [t for t, _ in paras]
        lines = lines or "(no results)"
        return f"Search results for '{query}':\n{lines}", [t for t, _ in paras]

    def gold_titles(self, supporting_facts):
        return {normalize_answer(t) for t in supporting_facts["title"]}


def parse_trajectory(text_turns):
    """把每个 assistant 轮分类为 search / answer / invalid，并提取内容"""
    parsed = []
    for raw in text_turns:
        m = ANSWER_RE.search(raw)
        if m:
            parsed.append(("answer", m.group(1).strip(), raw))
            continue
        m = SEARCH_RE.search(raw)
        if m:
            parsed.append(("search", m.group(1).strip(), raw))
            continue
        parsed.append(("invalid", None, raw))
    return parsed


def compute_reward(
    text_turns,           # 每个 assistant 轮的原始文本（按顺序）
    gold_answer: str,
    gold_title_set: set,  # gold 支撑段 title（原始形式，函数内部归一化）
    retrieved_titles: list,  # 每次检索命中的 title 列表（与 search 轮一一对应）
    max_search: int = 3,
    answer_metric: str = "f1",   # "f1"（软，起步）或 "em"（硬，收敛后）
    gate_process: bool = True,   # 过程奖励仅在合法 <answer> 终止时发放（防搜索循环坍缩）
    w_answer: float = 1.0,
    w_format: float = 0.2,
    w_process: float = 0.15,
    w_over: float = 0.2,
    max_turns: int = 4,
):
    parsed = parse_trajectory(text_turns)
    info = {}

    # ---- R_answer ----
    final_answer = None
    for kind, content, _ in reversed(parsed):
        if kind == "answer":
            final_answer = content
            break
    if final_answer is None:
        r_answer = 0.0
    elif answer_metric == "em":
        r_answer = exact_match(final_answer, gold_answer)
    else:
        r_answer = ans_f1(final_answer, gold_answer)

    # ---- R_format：所有轮合法且以 answer 收尾 ----
    all_valid = all(k in ("search", "answer") for k, _, _ in parsed)
    ends_answer = bool(parsed) and parsed[-1][0] == "answer"
    only_one_answer = sum(1 for k, _, _ in parsed if k == "answer") <= 1
    r_format = float(all_valid and ends_answer and only_one_answer)

    # ---- R_process：检索命中 gold 支撑段比例（两侧统一 normalize_answer）----
    # 关键设计（run1 崩溃教训）：过程奖励必须由合法终止门控。
    # 不门控时"多搜不答"轨迹可白拿过程分，GRPO 梯度会把策略推向搜索循环
    # （实测：一轮更新后 fmt 20.1%→0.2%，avg_search 3.8→3.99，作答行为灭绝）。
    gold_norm = {normalize_answer(t) for t in gold_title_set}
    if gate_process and r_format < 1.0:
        r_process = 0.0
    elif gold_norm and retrieved_titles:
        hit = set()
        for titles in retrieved_titles:
            hit |= {normalize_answer(t) for t in titles}
        r_process = len(hit & gold_norm) / len(gold_norm)
    else:
        r_process = 0.0

    # ---- R_over_retrieval：超出次数 / 轮数超限 ----
    n_search = sum(1 for k, _, _ in parsed if k == "search")
    over = max(0, n_search - max_search) + max(0, len(parsed) - max_turns)
    r_over = float(min(over, 2)) / 2.0

    total = (w_answer * r_answer + w_format * r_format + w_process * r_process
             - w_over * r_over)
    info.update({
        "r_answer": round(r_answer, 4), "r_format": r_format,
        "r_process": round(r_process, 4), "r_over": r_over,
        "n_search": n_search, "final_answer": final_answer,
        "total": round(total, 4),
    })
    return total, info


def grpo_advantages(rewards: list, group_size: int, eps: float = 1e-6):
    """组内归一化优势（GRPO）：rewards 按 prompt 顺序、每 group_size 个一组"""
    rewards = np.asarray(rewards, dtype=np.float32)
    adv = np.zeros_like(rewards)
    for i in range(0, len(rewards), group_size):
        g = rewards[i:i + group_size]
        std = g.std()
        adv[i:i + group_size] = (g - g.mean()) / (std + eps)
    return adv
