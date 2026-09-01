"""M1 基线标定：HotpotQA dev (distractor) 上测两条基线

基线 A（下界）：无检索，直接作答（纯参数化知识）
基线 B（上界参照）：给定 10 个候选段落（含 2 个 gold），直接作答（阅读理解上限）

指标：官方 HotpotQA EM / F1（normalize_answer）。
数据：data/hotpot_dev_distractor.parquet（HF hotpotqa/hotpot_qa, distractor, validation）

用法（宿主机）：
docker run --rm --gpus '"device=0"' \
  -v /mnt/nas3/shared/model:/models:ro \
  -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
  -e HF_HUB_OFFLINE=1 \
  1cat-vllm:v100-1.3.0 python m1_baseline.py --n 500
"""
import argparse
import json
import random
import re
import string
from collections import Counter

import pandas as pd
from vllm import LLM, SamplingParams

MODEL = "/models/Qwen3-4B"
DATA = "data/hotpot_dev_distractor.parquet"
OUT = "logs/m1_baseline_results.json"


# ---------- HotpotQA 官方 EM / F1 ----------
def normalize_answer(s):
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    nsame = sum(common.values())
    if nsame == 0:
        return 0.0
    return 2 * nsame / (len(p) + len(g))


def clean_pred(s):
    """去掉模型自行输出的 <think>...</think> 块（Qwen3 base 在非思考模式下也会回显空 think 标签）"""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    return s.strip()


# ---------- prompts ----------
# 注意：Qwen3 base 默认进入 <think> 长推理模式，实测会答不完就被截断导致 EM 全 0。
# 必须显式 /no_think 关闭 thinking；apply_chat_template 的 enable_thinking 参数在该模板下不生效。
NO_THINK = "/no_think"
SYS_DIRECT = f"{NO_THINK}\nYou are a helpful assistant. Answer the question concisely with a short phrase (at most 10 words). Do not explain."
SYS_RAG = f"{NO_THINK}\nYou are a helpful assistant. Answer the question based on the given context, concisely with a short phrase (at most 10 words). Do not explain."


def direct_prompt(q):
    return f"Question: {q}\nAnswer:"


def rag_prompt(q, ctx_paras):
    ctx = "\n".join(f"{title}: {''.join(sents)}" for title, sents in ctx_paras)
    return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(DATA)
    df = df[df["type"] == "bridge"]  # 多跳桥接题才是重点
    rows = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"loaded {len(df)} bridge dev rows, sampling {len(rows)}")

    qs = rows["question"].tolist()
    golds = rows["answer"].tolist()
    ctxs = []
    for c in rows["context"]:
        ctxs.append([(c["title"][i], c["sentences"][i]) for i in range(len(c["title"]))])

    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.6,
              max_model_len=4096, enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=48, temperature=0.0)

    def batch_gen(sysmsg, usermsgs):
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": sysmsg}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True,
            )
            for u in usermsgs
        ]
        return [o.outputs[0].text.strip() for o in llm.generate(prompts, sp)]

    print("== Baseline A: direct (no retrieval) ==")
    pred_a = batch_gen(SYS_DIRECT, [direct_prompt(q) for q in qs])
    print("== Baseline B: gold-context RAG ==")
    pred_b = batch_gen(SYS_RAG, [rag_prompt(q, c) for q, c in zip(qs, ctxs)])

    res = {}
    pred_a = [clean_pred(p) for p in pred_a]
    pred_b = [clean_pred(p) for p in pred_b]
    for name, preds in [("A_direct", pred_a), ("B_gold_rag", pred_b)]:
        em = sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(golds)
        f1v = sum(f1(p, g) for p, g in zip(preds, golds)) / len(golds)
        res[name] = {"EM": round(em * 100, 2), "F1": round(f1v * 100, 2)}
        print(f"{name}: EM={em*100:.2f}  F1={f1v*100:.2f}")

    res["meta"] = {"n": len(rows), "model": MODEL, "type": "bridge", "seed": args.seed}
    res["samples"] = [
        {"q": qs[i], "gold": golds[i], "A": pred_a[i], "B": pred_b[i]}
        for i in range(min(10, len(qs)))
    ]
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nsaved to {OUT}")
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
