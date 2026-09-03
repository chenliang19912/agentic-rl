"""中文迁移验证子实验 · 第二步：翻译结果锚定校验 + 标题规范化

教师模型翻译的文章标题必须能对应到中文维基语料中真实存在的文章，否则该问题
无法在我们的检索环境里复现多跳证据链。

锚定规则（逐级放宽）：
1. exact：归一化后精确匹配；
2. paren：去括号消歧后缀（"X（Y）" → "X"）再匹配；
3. retrieval：译名变体锚定——用译名在 index_zh 里检索，候选标题与译名的
   字 bigram 相似度达标即采纳（中文维基译名常有变体，如 里希特霍芬/李希霍芬，
   字面不同但 bigram 高度重叠）。
   已放弃"译名包含标题子串"的宽松规则：抽检发现误配率极高
   （迈克尔H·乔丹→迈克尔、NextGen America→NeXT 等）。
   retrieval 锚定残余的误配风险由下游"答案证据核验"门控兜底。

关键输出：titles_zh_canon —— 语料中真实存在的规范标题（非教师译名），
下游 SFT 覆盖检查、RL 过程奖励、评测 gold 一律用它，保证口径一致。

用法：
  python zh_anchor.py                 # 仅 exact+paren（不依赖索引）
  python zh_anchor.py --idx index_zh  # 加 retrieval 锚定

输出：data/zh_anchored_{dev,train}.jsonl（仅两个标题均锚定的样本）
"""
import argparse
import json
import re

from m1_baseline_c import BM25, normalize_answer
from textproc import has_cjk

RAW = "data/zhwiki_paras.jsonl"
SETS = [("dev", "data/zh_translated_dev.jsonl", "data/zh_anchored_dev.jsonl"),
        ("train", "data/zh_translated_train.jsonl", "data/zh_anchored_train.jsonl")]

_PAREN_RE = re.compile(r"[（(][^（()）]*[）)]$")


def strip_paren(t):
    prev = None
    while prev != t:
        prev = t
        t = _PAREN_RE.sub("", t).strip()
    return t


def char_bigrams(s):
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}


def bigram_sim(a, b):
    """返回 (dice, recall_b)：dice 对称相似度；recall_b = 标题 bigram 被译名覆盖比例"""
    A, B = char_bigrams(a), char_bigrams(b)
    if not A or not B:
        return 0.0, 0.0
    inter = len(A & B)
    return 2 * inter / (len(A) + len(B)), inter / len(B)


class TitleAnchor:
    def __init__(self, idx_dir=None):
        self.norm2orig = {}
        n = 0
        for line in open(RAW):
            obj = json.loads(line)
            key = normalize_answer(obj["title"])
            if key not in self.norm2orig:
                self.norm2orig[key] = obj["title"]
            n += 1
            if n % 2000000 == 0:
                print(f"scanned {n:,} paras ...", flush=True)
        self.title_keys = set(self.norm2orig)
        print(f"title set: {len(self.title_keys):,} unique", flush=True)
        self.bm25 = BM25(idx_dir) if idx_dir else None

    def _retrieval_anchor(self, q):
        """用译名检索语料，按 bigram 相似度决定是否采纳候选标题"""
        hits = self.bm25.search(q, k=5)
        best, best_score = None, 0.0
        for d in hits:
            cand = self.bm25.get_doc(d)[0]
            ck = normalize_answer(cand)
            if not has_cjk(ck):
                continue
            dice, recall_b = bigram_sim(q, ck)
            score = max(dice, recall_b)
            if score >= 0.6 and score > best_score:
                best, best_score = cand, score
        if best is not None:
            return best, f"retr({best_score:.2f})"
        return None, "miss"

    def anchor(self, title_zh):
        """返回 (规范标题, 方式) 或 (None, 'miss')"""
        q = normalize_answer(title_zh)
        if q in self.title_keys:
            return self.norm2orig[q], "exact"
        qs = strip_paren(q)
        if qs != q and qs in self.title_keys:
            return self.norm2orig[qs], "paren"
        if self.bm25 is not None:
            return self._retrieval_anchor(q)
        return None, "miss"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", default=None)
    args = ap.parse_args()

    ta = TitleAnchor(args.idx)
    for kind, src, dst in SETS:
        try:
            rows = [json.loads(l) for l in open(src)]
        except FileNotFoundError:
            print(f"-- skip {kind}: {src} not found")
            continue
        rows = [r for r in rows if r.get("parse_ok")]
        anchored, methods, samples = [], {}, []
        for r in rows:
            c0, m0 = ta.anchor(r["titles_zh"][0])
            c1, m1 = ta.anchor(r["titles_zh"][1])
            if c0 is None or c1 is None:
                if len(samples) < 5:
                    samples.append({"titles_zh": r["titles_zh"],
                                    "hit": [c0 is not None, c1 is not None]})
                continue
            for m in (m0, m1):
                key = m.split("(")[0]
                methods[key] = methods.get(key, 0) + 1
            anchored.append({**r, "titles_zh_canon": [c0, c1]})
        n = len(rows)
        print(f"== {kind}: parse_ok={n} anchored={len(anchored)} "
              f"({len(anchored)/max(n,1)*100:.1f}%)  title_methods={methods}")
        for s in samples:
            print("  miss:", json.dumps(s, ensure_ascii=False))
        if anchored:
            print("  sample anchored:", json.dumps(
                {"q": anchored[0]["question_zh"],
                 "canon": anchored[0]["titles_zh_canon"],
                 "trans": anchored[0]["titles_zh"]}, ensure_ascii=False))
        with open(dst, "w") as f:
            for r in anchored:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  saved -> {dst}")


if __name__ == "__main__":
    main()
