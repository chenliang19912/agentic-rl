"""下载中文维基百科（hf-mirror 的 wikimedia/wikipedia 20231101.zh）并切段

产出与 wiki-18 同构的段落语料 data/zhwiki_paras.jsonl：
每行 {"id": "zhwiki-<n>", "title": 文章标题, "text": 段落文本}
切段策略：按空行/章节切块，超过 ~200 字的块按句子再切，目标段长 100–250 字。

用法：
docker run --rm -v ...:/work -w /work -e HF_ENDPOINT=https://hf-mirror.com \
  1cat-vllm:v100-1.3.0 python -u zh_wiki_download.py
"""
import json
import os
import re

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/work/hf_cache_zh")

from datasets import load_dataset  # noqa: E402

RAW = "data/zhwiki_raw.jsonl"
OUT = "data/zhwiki_paras.jsonl"
MAX_CHARS = 250
MIN_CHARS = 40

SECTION_RE = re.compile(r"^==+.*?==+\s*$")
SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])")


def split_article(text):
    """按空行与章节行切块，再按句把长块切到目标长度"""
    blocks = []
    for blk in re.split(r"\n\s*\n", text):
        lines = [ln for ln in blk.split("\n") if not SECTION_RE.match(ln)]
        blk = "\n".join(lines).strip()
        if blk:
            blocks.append(blk)
    paras = []
    for blk in blocks:
        if len(blk) <= MAX_CHARS:
            paras.append(blk.replace("\n", " "))
            continue
        # 长块按句累积到 ~200 字
        sents = [s for s in SENT_SPLIT_RE.split(blk) if s.strip()]
        buf = ""
        for s in sents:
            if len(buf) + len(s) > MAX_CHARS and buf:
                paras.append(buf.replace("\n", " ").strip())
                buf = ""
            buf += s
        if buf.strip():
            paras.append(buf.replace("\n", " ").strip())
    return [p for p in paras if len(p) >= MIN_CHARS]


def main():
    # ---- 1. 下载（缓存分片，可断点续传）----
    if not os.path.exists(RAW):
        ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")
        print(f"downloaded {len(ds)} articles", flush=True)
        with open(RAW, "w") as f:
            for i, ex in enumerate(ds):
                f.write(json.dumps({"id": ex["id"], "title": ex["title"],
                                    "text": ex["text"]}, ensure_ascii=False) + "\n")
                if (i + 1) % 100000 == 0:
                    print(f"raw dumped {i + 1} ...", flush=True)
    else:
        print(f"reuse existing {RAW}", flush=True)

    # ---- 2. 切段 ----
    n_paras, n_skipped = 0, 0
    with open(OUT, "w") as f:
        for line in open(RAW):
            art = json.loads(line)
            paras = split_article(art["text"])
            for j, p in enumerate(paras):
                f.write(json.dumps({"id": f"zhwiki-{n_paras}",
                                    "title": art["title"], "text": p},
                                   ensure_ascii=False) + "\n")
                n_paras += 1
            if not paras:
                n_skipped += 1
            if n_paras and n_paras % 1000000 == 0:
                print(f"paras {n_paras} ...", flush=True)
    print(json.dumps({"paras": n_paras, "articles_no_para": n_skipped}))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
