"""M5 自适应检索 k 环境 + 奖励（A1 检索动作联合优化；协议与判据见 M5_ADAPTIVE_K.md，
奖励与 m3_env 完全同权重，仅动作空间扩展）

协议设计（防坍缩优先）：
- k 为**可选**参数：<search>query</search> 依旧合法（等价 k=6），<search k=8>query</search>
  请求 k 条结果。SFT 冷启动模型从未见过 k 语法，强制新语法会有格式坍缩风险
  （run1 教训）；可选设计让 k 使用成为 RL 自主发现的行为。
- k ∈ 整数 1..12，越界钳制到边界（环境侧防御，不惩罚——动作空间定义的一部分）。
- 奖励函数不动：R = F1 + 0.2·格式 + 0.15·门控过程 − 0.2·超限（可验证奖励不变，
  k 选择的收益只能通过 F1/过程分间接体现——这正是"联合优化"要检验的）。

M5 预注册判据（摘要）：贪心 EM 相对 run3-iter0（26.60）非劣（±1.0pt）且检索结果
字符量下降 ≥10%；或 EM +≥2.0pt 且搜索次数不升。k 分布退化为常数 = "未涌现"，
如实报告。
"""
import re

from m1_baseline_c import BM25, normalize_answer
from m3_env import (ANSWER_RE, compute_reward as _compute_reward_m3,
                    grpo_advantages)

MAX_SEARCH = 3
MAX_TURNS = 4
K_DEFAULT = 6
K_MIN, K_MAX = 1, 12

SYSTEM_M5 = (
    "You are a question-answering agent. Answer the question using search results.\n"
    "To search, output exactly: <search>your query</search> (returns 6 results by default),\n"
    "or request a different number of results: <search k=8>your query</search> (k = 1 to 12).\n"
    "When you can answer, output exactly: <answer>your answer</answer>\n"
    "Search at most 3 times. Answer must be concise."
)

# k 捕获：<search k=8> / <search k = 8> / <search> 均可
SEARCH_RE_M5 = re.compile(r"<search(?:\s*k\s*=\s*(\d+))?\s*>(.*?)(?:</search>|$)", re.S)


class SearchEnvM5:
    """自适应 k 的 BM25 检索环境。接口与 m3_env.SearchEnv 对齐，search 增加逐次 k。"""

    def __init__(self, idx_dir="index", lang="en"):
        self.lang = lang
        self.bm25 = BM25(idx_dir)

    def search(self, query: str, k=None):
        """k=None/非法 → 默认 6；否则钳制到 [K_MIN, K_MAX]"""
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = K_DEFAULT
        k = max(K_MIN, min(K_MAX, k))
        hits = self.bm25.search(query, k=k)
        paras = [self.bm25.get_doc(d) for d in hits]
        lines = "\n".join(f"- {t}: {x}" for t, x in paras)
        lines = lines or "(no results)"
        text = f"Search results for '{query}':\n{lines}"
        return text, [t for t, _ in paras], k


def parse_trajectory_m5(text_turns):
    """[(kind, content, k, raw)]：k 仅 search 轮有值（None=未指定）"""
    out = []
    for raw in text_turns:
        m = ANSWER_RE.search(raw)
        if m:
            out.append(("answer", m.group(1).strip(), None, raw))
            continue
        m = SEARCH_RE_M5.search(raw)
        if m:
            out.append(("search", m.group(2).strip(), m.group(1), raw))
            continue
        out.append(("invalid", None, None, raw))
    return out


def compute_reward_m5(text_turns, gold_answer, gold_title_set, retrieved_titles,
                      search_chars=None, **kw):
    """奖励语义与 m3_env.compute_reward 完全一致；额外统计 k 使用与检索字符量。
    实现：把 k= 前缀从轮文本剥掉后复用 m3 原函数（单一事实来源，避免权重漂移），
    再叠加 k/字符统计。"""
    stripped = [SEARCH_RE_M5.sub(lambda m: f"<search>{m.group(2)}</search>", t)
                for t in text_turns]
    reward, info = _compute_reward_m3(stripped, gold_answer, gold_title_set,
                                      retrieved_titles, **kw)
    parsed = parse_trajectory_m5(text_turns)
    k_values = [int(k) if k is not None else K_DEFAULT
                for kind, _, k, _ in parsed if kind == "search"]
    info["k_values"] = k_values
    info["k_specified_pct"] = round(
        sum(1 for kind, _, k, _ in parsed if kind == "search" and k is not None)
        / max(1, len(k_values)), 4)
    info["avg_k"] = round(sum(k_values) / len(k_values), 2) if k_values else 0.0
    info["search_chars"] = sum(search_chars) if search_chars else 0
    return reward, info
