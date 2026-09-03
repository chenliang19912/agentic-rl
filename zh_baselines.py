"""中文迁移验证子实验 · 基线：无检索直答 / BM25 真实 RAG（中文评测集）

与英文主线的 Baseline A/C 同构：
- A_zh：无检索，模型直接回答（测语言内参数化知识）
- C_zh：BM25 index_zh top-k 检索后作答（测"检索到位但不会多跳"的水平）
评测集：data/zh_anchored_dev.jsonl（翻译+锚定后的 HotpotQA dev bridge）。
中文 EM/F1：字符级（m1_baseline_c 已适配）。

用法：
docker run --rm --gpus '"device=0"' \
  -v /mnt/nas3/shared/model:/models:ro \
  -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
  -e HF_HUB_OFFLINE=1 \
  1cat-vllm:v100-1.3.0 python -u zh_baselines.py
"""
import json

from vllm import LLM, SamplingParams

from m1_baseline_c import exact_match, f1
from m3_env import SearchEnv

MODEL = "/models/Qwen3-4B"
DEV = "data/hotpot_zh_dev.jsonl"   # 门控过滤后的"语料内可答"评测集
OUT = "logs/zh_baselines_results.json"

SYS_DIRECT = "/no_think\n你是一个知识渊博的助手。请简短回答问题（最多 10 个字），不要解释。"
SYS_RAG = "/no_think\n你是一个知识渊博的助手。请根据给出的上下文简短回答问题（最多 10 个字），不要解释。"


def direct_prompt(q):
    return f"问题：{q}\n答案："


def rag_prompt(q, paras):
    ctx = "\n".join(f"{t}：{x}" for t, x in paras)
    return f"上下文：\n{ctx}\n\n问题：{q}\n答案："


def main():
    rows = [json.loads(l) for l in open(DEV)]
    print(f"zh eval set: {len(rows)} anchored dev items")
    qs = [r["question_zh"] for r in rows]
    golds = [r["answer_zh"] for r in rows]

    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.6,
              max_model_len=4096, enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=48, temperature=0.0)

    def gen(sys_prompt, user_prompts):
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True) for u in user_prompts]
        return [o.outputs[0].text.strip() for o in llm.generate(prompts, sp)]

    res = {"meta": {"n": len(rows), "model": MODEL, "source": DEV}}

    preds = gen(SYS_DIRECT, [direct_prompt(q) for q in qs])
    res["A_zh_direct"] = {
        "EM": round(sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(qs) * 100, 2),
        "F1": round(sum(f1(p, g) for p, g in zip(preds, golds)) / len(qs) * 100, 2),
    }
    print("A_zh_direct:", json.dumps(res["A_zh_direct"], ensure_ascii=False))

    env = SearchEnv(top_k=10, idx_dir="index_zh", lang="zh")
    paras_all = [[env.bm25.get_doc(d) for d in env.bm25.search(q, k=10)]
                 for q in qs]
    preds = gen(SYS_RAG, [rag_prompt(q, p) for q, p in zip(qs, paras_all)])
    res["C_zh_bm25_rag"] = {
        "EM": round(sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(qs) * 100, 2),
        "F1": round(sum(f1(p, g) for p, g in zip(preds, golds)) / len(qs) * 100, 2),
    }
    print("C_zh_bm25_rag:", json.dumps(res["C_zh_bm25_rag"], ensure_ascii=False))

    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
