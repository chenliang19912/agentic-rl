"""M1 基线 C：真实 RAG（BM25 wiki-18 全量检索 top-k 后作答）

与基线 B（oracle gold 上下文）的区别：这里的上下文由检索器给出，
检索质量成为约束 —— 这是 RL 训练中检索环境要逼近的真实上限。

附带检索质量诊断：gold 段落 recall@k（用 supporting_facts 的 title 匹配）。

协议与 m1_baseline.py 完全一致：HotpotQA dev(distractor) bridge n=500 seed=42，
/no_think + 简洁短答，官方 EM/F1（clean_pred 去 think 标签）。

用法（宿主机）：
docker run --rm --gpus '"device=0"' \
  -v /mnt/nas3/shared/model:/models:ro \
  -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
  -e HF_HUB_OFFLINE=1 \
  1cat-vllm:v100-1.3.0 python m1_baseline_c.py --n 500
"""
import argparse
import json
import math
import pickle
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams

MODEL = "/models/Qwen3-4B"
IDX = "index"
DATA = "data/hotpot_dev_distractor.parquet"
OUT = "logs/m1_baseline_c_results.json"

K1, B = 1.5, 0.75
MAX_DF_RATIO = 0.01  # df 超过语料 1% 的词视为停用词跳过

from textproc import tokenize  # 英文按词 / 中文字 bigram，与建索引端同一实现  # noqa: E402


# ---------- HotpotQA 官方 EM / F1 ----------
def normalize_answer(s):
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


_ZH_PUNC = set("，。！？；：、""''（）《》【】—…·「」『』〈〉")


def _has_cjk(s):
    return any("一" <= ch <= "鿿" for ch in s)


def _zh_chars(s):
    return [ch for ch in s.lower()
            if ch not in string.punctuation and ch not in _ZH_PUNC and not ch.isspace()]


def exact_match(pred, gold):
    if _has_cjk(pred) or _has_cjk(gold):   # 中文按字符级（社区标准口径）
        return float(_zh_chars(pred) == _zh_chars(gold))
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred, gold):
    if _has_cjk(pred) or _has_cjk(gold):
        p, g = _zh_chars(pred), _zh_chars(gold)
    else:
        p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    nsame = sum(common.values())
    if nsame == 0:
        return 0.0
    return 2 * nsame / (len(p) + len(g))


def clean_pred(s):
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    return s.strip()


# ---------- BM25 检索器 ----------
class BM25:
    def __init__(self, idx_dir=IDX):
        with open(f"{idx_dir}/vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)
        self.term_ptr = np.load(f"{idx_dir}/term_ptr.npy")
        self.doc_ids = np.load(f"{idx_dir}/doc_ids.npy", mmap_mode="r")
        self.tfs = np.load(f"{idx_dir}/tfs.npy", mmap_mode="r")
        self.doc_len = np.load(f"{idx_dir}/doc_len.npy")
        self.offsets = np.load(f"{idx_dir}/offsets.npy")
        self.N = len(self.doc_len)
        self.AVGDL = float(self.doc_len.mean())
        self.scores = np.zeros(self.N, dtype=np.float32)
        self.fcorpus = open(f"{idx_dir}/corpus.jsonl", "rb")
        print(f"BM25 loaded: N={self.N:,} V={len(self.vocab):,} avgdl={self.AVGDL:.1f}")

    def search(self, query, k=10):
        scores = self.scores
        touched = []
        for term in set(tokenize(query)):
            tid = self.vocab.get(term)
            if tid is None:
                continue
            lo, hi = int(self.term_ptr[tid]), int(self.term_ptr[tid + 1])
            df = hi - lo
            if df == 0 or df > self.N * MAX_DF_RATIO:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            ds = np.asarray(self.doc_ids[lo:hi])
            ts = np.asarray(self.tfs[lo:hi], dtype=np.float32)
            dls = self.doc_len[ds]
            s = idf * (ts * (K1 + 1)) / (ts + K1 * (1 - B + B * dls / self.AVGDL))
            np.add.at(scores, ds, s)
            touched.append(ds)
        if not touched:
            return []
        cand = np.unique(np.concatenate(touched))
        cs = scores[cand]
        kk = min(k, len(cand))
        top = np.argpartition(-cs, kk - 1)[:kk]
        top = top[np.argsort(-cs[top])]
        result = cand[top].tolist()
        scores[cand] = 0.0
        return result

    def get_doc(self, doc_id):
        """返回 (title, text)。兼容两种语料格式：
        wiki-18 的 {"id","contents": "title\\n正文"} 与
        zhwiki 的 {"id","title","text"} 分离字段"""
        self.fcorpus.seek(int(self.offsets[doc_id]))
        obj = json.loads(self.fcorpus.readline())
        contents = obj.get("contents")
        if contents is None:
            return obj.get("title", "").strip(), obj.get("text", "").strip()
        title, _, text = contents.partition("\n")
        return title.strip().strip('"'), text.strip()


# ---------- prompts ----------
NO_THINK = "/no_think"
SYS_RAG = f"{NO_THINK}\nYou are a helpful assistant. Answer the question based on the given context, concisely with a short phrase (at most 10 words). Do not explain."


def rag_prompt(q, paras):
    ctx = "\n".join(f"{t}: {x}" for t, x in paras)
    return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_parquet(DATA)
    df = df[df["type"] == "bridge"]
    rows = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"loaded {len(df)} bridge dev rows, sampling {len(rows)}")

    qs = rows["question"].tolist()
    golds = rows["answer"].tolist()
    gold_titles = []
    for sf in rows["supporting_facts"]:
        gold_titles.append({normalize_answer(t) for t in sf["title"]})

    bm25 = BM25()

    # ---- 检索 + 诊断 ----
    print(f"retrieving top-{args.k} for {len(qs)} questions...")
    paras_all, rec1, rec_all, rec_none = [], 0.0, 0, 0
    for i, q in enumerate(qs):
        hits = bm25.search(q, k=args.k)
        paras = [bm25.get_doc(d) for d in hits]
        paras_all.append(paras)
        got = {normalize_answer(t) for t, _ in paras}
        inter = len(got & gold_titles[i])
        rec1 += inter / max(len(gold_titles[i]), 1)
        rec_all += int(inter == len(gold_titles[i]) and inter > 0)
        rec_none += int(inter == 0)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(qs)} retrieved", flush=True)
    n = len(qs)
    recall = {
        "gold_para_recall_at_k": round(rec1 / n, 4),
        "all_gold_hit_pct": round(rec_all / n * 100, 2),
        "no_gold_hit_pct": round(rec_none / n * 100, 2),
        "k": args.k,
    }
    print("retrieval diagnostics:", json.dumps(recall))

    # ---- 生成 ----
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.6,
              max_model_len=4096, enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=48, temperature=0.0)
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYS_RAG},
             {"role": "user", "content": rag_prompt(q, paras)}],
            tokenize=False, add_generation_prompt=True,
        )
        for q, paras in zip(qs, paras_all)
    ]
    preds = [clean_pred(o.outputs[0].text.strip()) for o in llm.generate(prompts, sp)]

    em = sum(exact_match(p, g) for p, g in zip(preds, golds)) / n
    f1v = sum(f1(p, g) for p, g in zip(preds, golds)) / n
    res = {"C_bm25_rag": {"EM": round(em * 100, 2), "F1": round(f1v * 100, 2)},
           "retrieval": recall,
           "meta": {"n": n, "model": MODEL, "type": "bridge", "seed": args.seed}}
    res["samples"] = [
        {"q": qs[i], "gold": golds[i], "pred": preds[i],
         "ret_titles": [t for t, _ in paras_all[i][:3]]}
        for i in range(min(10, n))
    ]
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n== Baseline C: BM25 top-{args.k} RAG ==")
    print(f"EM={em*100:.2f}  F1={f1v*100:.2f}")
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, ensure_ascii=False))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
