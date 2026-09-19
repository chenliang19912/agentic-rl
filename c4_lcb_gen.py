"""C4 LCB held-out · Phase A：批量生成候选并落盘（docker 内单卡 GPU）

n=32/题，seed=42，候选经 normalize_candidate 归一化（class→裸函数）后连同
原始 fence 提取码全部写入 candidates.jsonl——验证阶段（Phase B）独立并行跑，
崩溃不丢生成结果。

用法（docker 内）：python c4_lcb_gen.py --model /models/Qwen3-4B-Instruct-2507
输出 data/lcb/candidates.jsonl：{tid, fn_name, difficulty, cands: [{code, class_stripped}]}
"""
import argparse
import json

from vllm import LLM, SamplingParams

from c4_lcb_refgen import FENCE_RE, normalize_candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--pool", default="data/lcb/pool.jsonl")
    ap.add_argument("--out", default="data/lcb/candidates.jsonl")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool)]
    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=0.7,
              max_model_len=8192, enforce_eager=True)
    tok = llm.get_tokenizer()

    prompts, metas = [], []
    for p in pool:
        user = (p["question_content"]
                + f"\n\n[Format] Implement your solution as a plain Python function "
                  f"(do NOT wrap it in a class):\n{p['signature']}\n"
                  f"typing imports and common stdlib modules are already available. "
                  f"Return only the final code in a ```python fence.")
        msgs = [{"role": "system",
                 "content": "You are an expert competitive programmer. Write correct, efficient Python."},
                {"role": "user", "content": user}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
        metas.append(p)

    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)

    n_cands = 0
    with open(args.out, "w") as f:
        for p, o in zip(metas, outs):
            cands = []
            for cand in o.outputs:
                text = cand.text.strip()
                m = FENCE_RE.search(text)
                code = (m.group(1) if m else text).strip()
                if not code:
                    continue
                code, stripped = normalize_candidate(code, p["fn_name"])
                cands.append({"code": code, "class_stripped": stripped})
            n_cands += len(cands)
            f.write(json.dumps({"tid": p["tid"], "fn_name": p["fn_name"],
                                "difficulty": p["difficulty"],
                                "cands": cands}, ensure_ascii=False) + "\n")
            print(f"[gen] {p['tid']} cands={len(cands)}", flush=True)
    print(json.dumps({"pool": len(pool), "total_cands": n_cands,
                      "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
