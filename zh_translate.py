"""中文迁移验证子实验 · 第一步：教师模型翻译 HotpotQA bridge 题目

用 Qwen3-4B-Instruct-2507（教师）把 HotpotQA bridge 的 question / answer /
两个支撑文章 title 翻译成简体中文，供后续中文 SFT/RL/评测使用。

题目选取：
- dev：与英文主评测完全同题（dev bridge, seed=42, n=500）→ 语言间可直接对比
- train：train bridge 采样 6000（给锚定过滤与 SFT/RL 划分留足余量）

输出：data/zh_translated_dev.jsonl / data/zh_translated_train.jsonl
每行：{"question","answer","titles","question_zh","answer_zh","titles_zh","parse_ok"}

用法：
docker run --rm --gpus '"device=5"' \
  -v /mnt/nas3/shared/model:/models:ro \
  -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
  -e HF_HUB_OFFLINE=1 \
  1cat-vllm:v100-1.3.0 python -u zh_translate.py
"""
import argparse
import json
import re

import pandas as pd
from vllm import LLM, SamplingParams

MODEL = "/models/Qwen3-4B-Instruct-2507"

DEV_PARQUET = "data/hotpot_dev_distractor.parquet"
TRAIN_PARQUETS = ["data/hotpot_train_0.parquet", "data/hotpot_train_1.parquet"]

SYS = (
    "/no_think\n你是专业翻译。将英文维基百科多跳问答题翻译成简体中文。要求：\n"
    "1. 人名、地名、作品名等专有名词使用中文维基百科最常用的译名；\n"
    "2. 问题保持疑问句；答案保持简短短语；\n"
    "3. 文章标题翻译成中文维基风格的文章标题，不加书名号或其他标点；\n"
    "4. 只输出一个 JSON 对象，不要输出任何其他内容，格式：\n"
    '{"question_zh": "...", "answer_zh": "...", "titles_zh": ["...", "..."]}'
)

USER_TMPL = "Question: {q}\nAnswer: {a}\nSupporting titles: {t1} | {t2}"

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_translation(text):
    """从模型输出中稳健地解析 JSON；失败返回 None"""
    text = text.strip()
    if text.startswith("```"):  # 容错：模型偶尔包 markdown 代码块
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    tz = obj.get("titles_zh")
    if not (obj.get("question_zh") and obj.get("answer_zh")
            and isinstance(tz, list) and len(tz) == 2
            and all(isinstance(t, str) and t.strip() for t in tz)):
        return None
    return {"question_zh": str(obj["question_zh"]).strip(),
            "answer_zh": str(obj["answer_zh"]).strip(),
            "titles_zh": [t.strip() for t in tz]}


def translate_batch(llm, tok, rows):
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": USER_TMPL.format(
                 q=r["question"], a=r["answer"],
                 t1=r["titles"][0], t2=r["titles"][1])}],
            tokenize=False, add_generation_prompt=True)
        for r in rows
    ]
    sp = SamplingParams(max_tokens=256, temperature=0.0)
    outs = llm.generate(prompts, sp)
    results = []
    for r, o in zip(rows, outs):
        parsed = parse_translation(o.outputs[0].text)
        item = {"question": r["question"], "answer": r["answer"],
                "titles": r["titles"]}
        if parsed:
            item.update(parsed)
            item["parse_ok"] = True
        else:
            item.update({"question_zh": "", "answer_zh": "",
                         "titles_zh": [], "parse_ok": False,
                         "raw": o.outputs[0].text[:500]})
        results.append(item)
    return results


def load_split(kind, n, seed, exclude_qs=None):
    if kind == "dev":
        df = pd.read_parquet(DEV_PARQUET)
    else:
        df = pd.concat([pd.read_parquet(p) for p in TRAIN_PARQUETS],
                       ignore_index=True)
    df = df[df["type"] == "bridge"].reset_index(drop=True)
    if exclude_qs:
        df = df[~df["question"].isin(exclude_qs)]
    if n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        titles = list(r["supporting_facts"]["title"])
        if len(titles) != 2:
            continue  # bridge 题标准是两个支撑段
        rows.append({"question": r["question"], "answer": r["answer"],
                     "titles": titles})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["dev", "train", "both"], default="both")
    ap.add_argument("--n_dev", type=int, default=500)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--seed_dev", type=int, default=42)   # 与英文主评测同
    ap.add_argument("--seed_train", type=int, default=31)
    ap.add_argument("--exclude", default=None,
                    help="已翻译结果文件：剔除其中问题，避免重复翻译")
    ap.add_argument("--append", action="store_true",
                    help="追加写出（与 --exclude 配合做增量翻译）")
    args = ap.parse_args()

    exclude_qs = None
    if args.exclude:
        exclude_qs = {json.loads(l)["question"] for l in open(args.exclude)}
        print(f"excluding {len(exclude_qs)} already-translated questions",
              flush=True)

    jobs = []
    if args.kind in ("dev", "both"):
        jobs.append(("dev", load_split("dev", args.n_dev, args.seed_dev, exclude_qs),
                     "data/zh_translated_dev.jsonl"))
    if args.kind in ("train", "both"):
        jobs.append(("train", load_split("train", args.n_train, args.seed_train, exclude_qs),
                     "data/zh_translated_train.jsonl"))

    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.5,
              max_model_len=1024, enforce_eager=True)
    tok = llm.get_tokenizer()

    for kind, rows, out_path in jobs:
        print(f"== translating {kind}: {len(rows)} items ==", flush=True)
        results = translate_batch(llm, tok, rows)
        ok = sum(1 for r in results if r["parse_ok"])
        with open(out_path, "a" if args.append else "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{kind}: parse_ok {ok}/{len(results)} -> {out_path}", flush=True)
        for r in results[:3]:
            print(json.dumps({k: r[k] for k in
                              ("question", "question_zh", "answer_zh", "titles_zh")
                              if r.get(k)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
