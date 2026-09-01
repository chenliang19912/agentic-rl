"""构建 wiki-18 BM25 倒排索引（CSR 格式）+ 原文偏移量表

输入：data/wiki-18.jsonl.gz（Search-R1 官方语料，实为 gz 压缩的 tar 包，
成员 wiki_dump.jsonl 约 2100 万维基段落，每行 {"id","contents"}，contents='"title"\n正文'）
输出（index/ 目录）：
  vocab.pkl      term -> tid 字典
  term_ptr.npy   int64  [V+1]  CSR 行指针
  doc_ids.npy    uint32 [P]   按 tid 排序后的文档号
  tfs.npy        uint16 [P]   对应词频
  doc_len.npy    uint32 [N]   每篇词数
  offsets.npy    uint64 [N]   每篇在 corpus.jsonl 中的字节偏移（随机取原文用）
  corpus.jsonl   解压后的语料（与输入同序），供检索后取回原文
  meta.json      N / V / P / avgdl 等

单进程，预计 30-60 分钟。用法：
docker run --rm -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
  1cat-vllm:v100-1.3.0 python build_bm25_index.py
"""
import argparse
import gzip
import json
import os
import pickle
import tarfile
import time

import numpy as np
from collections import Counter

from textproc import tokenize

SRC = "data/wiki-18.jsonl.gz"
OUT = "index"
CHUNK = 1 << 25  # 每个临时 chunk ~3300 万条 posting


def iter_lines(src, fmt):
    """tar：wiki-18 的 gz 套 tar 流式读；jsonl：普通逐行读（如 zhwiki_paras.jsonl）"""
    if fmt == "tar":
        with gzip.open(src, "rb") as gz, tarfile.open(fileobj=gz, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                for line in f:
                    yield line
    else:
        with open(src, "rb") as f:
            for line in f:
                yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--format", default="tar", choices=["tar", "jsonl"])
    args = ap.parse_args()
    src, out_dir, fmt = args.src, args.out, args.format

    os.makedirs(out_dir, exist_ok=True)
    vocab = {}
    doc_len = []
    offsets = []
    tids, dids, tfs = [], [], []
    n_docs = 0
    chunk_files = []
    out_offset = 0
    t0 = time.time()

    def flush():
        nonlocal tids, dids, tfs
        if not tids:
            return
        i = len(chunk_files)
        path = f"{out_dir}/_chunk_{i:03d}.npz"
        np.savez(
            path,
            tid=np.array(tids, dtype=np.uint32),
            did=np.array(dids, dtype=np.uint32),
            tf=np.array(tfs, dtype=np.uint16),
        )
        chunk_files.append(path)
        tids, dids, tfs = [], [], []

    fcorpus = open(f"{out_dir}/corpus.jsonl", "wb")
    for line in iter_lines(src, fmt):
        offsets.append(out_offset)
        fcorpus.write(line)
        out_offset += len(line)
        obj = json.loads(line)
        text = obj.get("contents") or (obj.get("title", "") + "\n" + obj.get("text", ""))
        toks = tokenize(text)
        doc_len.append(len(toks))
        for term, c in Counter(toks).items():
            tid = vocab.get(term)
            if tid is None:
                tid = len(vocab)
                vocab[term] = tid
            tids.append(tid)
            dids.append(n_docs)
            tfs.append(c)
        n_docs += 1
        if len(tids) >= CHUNK:
            flush()
        if n_docs % 1_000_000 == 0:
            el = time.time() - t0
            print(
                f"[{el:7.0f}s] docs={n_docs:,} vocab={len(vocab):,} "
                f"postings~={len(chunk_files) * CHUNK + len(tids):,}",
                flush=True,
            )
    fcorpus.close()
    flush()

    # ---- 合并所有 chunk ----
    arrays = [np.load(c) for c in chunk_files]
    tid_all = np.concatenate([a["tid"] for a in arrays])
    did_all = np.concatenate([a["did"] for a in arrays])
    tf_all = np.concatenate([a["tf"] for a in arrays])
    del arrays
    for c in chunk_files:
        os.remove(c)
    P = tid_all.shape[0]
    V = len(vocab)
    print(f"docs={n_docs:,} vocab={V:,} postings={P:,}; merged in {time.time()-t0:.0f}s")

    # CSR 行指针（先算，再动 tid_all）
    counts = np.bincount(tid_all, minlength=V)
    term_ptr = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    assert term_ptr[-1] == P
    del counts

    # ---- 按 (tid, did) 排序 ----
    print("sorting by (tid, did)...")
    keys = tid_all.astype(np.uint64) << 32 | did_all
    del tid_all
    order = np.argsort(keys, kind="stable")
    del keys
    did_sorted = did_all[order]
    tf_sorted = tf_all[order]
    del did_all, tf_all, order
    print(f"sorted in {time.time()-t0:.0f}s, saving...")

    # ---- 保存 ----
    np.save(f"{out_dir}/term_ptr.npy", term_ptr)
    np.save(f"{out_dir}/doc_ids.npy", did_sorted)
    np.save(f"{out_dir}/tfs.npy", tf_sorted)
    np.save(f"{out_dir}/doc_len.npy", np.array(doc_len, dtype=np.uint32))
    np.save(f"{out_dir}/offsets.npy", np.array(offsets, dtype=np.uint64))
    with open(f"{out_dir}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f, protocol=4)
    meta = {
        "N": n_docs,
        "V": V,
        "P": int(P),
        "avgdl": float(np.mean(doc_len)),
        "source": src,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
