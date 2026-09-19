"""M5 批量多轮搜索 rollout（自适应 k 版，m3_rollout 的协议扩展副本）

差异仅三处（训练层零改动）：
1. SYSTEM 用 m5_env.SYSTEM_M5（描述可选 k 动作空间）
2. 解析用 SEARCH_RE_M5，逐次检索把 k 传给环境（钳制 1..12，默认 6）
3. 轨迹额外记录 k_values / search_chars（检索成本计量的预注册次指标）

评测用法（与 m3_rollout 同 CLI）：
  python m5_rollout.py --model checkpoints/m5/iter0_merged --data eval --n 500 \
    --temperature 0 --system m5
"""
import argparse
import json

from vllm import LLM, SamplingParams

from m3_rollout import load_eval_questions, MAX_SEARCH, MAX_TURNS
from m5_env import (SYSTEM_M5, SearchEnvM5, compute_reward_m5,
                    parse_trajectory_m5)


def rollout(model_path, questions, G=1, max_seqs=256,
            temperature=1.0, dtype="float16", gpu_util=0.7, max_model_len=8192,
            idx_dir="index", system=None, max_new_tokens=200):
    llm = LLM(model=model_path, dtype=dtype, gpu_memory_utilization=gpu_util,
              max_model_len=max_model_len, enforce_eager=True)
    tok = llm.get_tokenizer()
    env = SearchEnvM5(idx_dir=idx_dir)
    sys_prompt = system or SYSTEM_M5
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                        top_p=1.0, stop=["</search>", "</answer>"])

    seqs = []
    for qi, item in enumerate(questions):
        for _ in range(G):
            seqs.append({
                "qi": qi,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Question: {item['question']}"},
                ],
                "turn_texts": [], "retrieved_titles": [], "search_chars": [],
                "done": False,
            })
    active = list(range(len(seqs)))

    for turn in range(MAX_TURNS):
        if not active:
            break
        prompts = [
            tok.apply_chat_template(seqs[i]["messages"], tokenize=False,
                                    add_generation_prompt=True)
            for i in active
        ]
        outs = []
        for b0 in range(0, len(prompts), max_seqs):
            outs += llm.generate(prompts[b0:b0 + max_seqs], sp)
        next_active = []
        for j, i in enumerate(active):
            raw = outs[j].outputs[0].text.strip()
            s = seqs[i]
            s["turn_texts"].append(raw)
            kind, content, k, _ = parse_trajectory_m5([raw])[0]
            if kind == "search" and len(s["retrieved_titles"]) < MAX_SEARCH:
                query = content or "search"
                result_text, titles, k_eff = env.search(query, k=k)
                s["retrieved_titles"].append(titles)
                s["search_chars"].append(len(result_text))
                # 规范化存储：k=None 的裸 search 补记为 <search>（与解析语义一致）；
                # 指定了 k 则按模型原始写法存储（钳制是环境侧行为，奖励管教）
                stored_k = f" k={k}" if k is not None else ""
                s["messages"].append({"role": "assistant",
                                      "content": f"<search{stored_k}>{query}</search>"})
                s["messages"].append({"role": "user", "content": result_text})
                next_active.append(i)
            else:
                if kind == "search":
                    stored = f"<search>{content or 'search'}</search>"
                elif kind == "answer":
                    stored = f"<answer>{content}</answer>"
                else:
                    stored = raw
                s["messages"].append({"role": "assistant", "content": stored})
                s["done"] = True
        active = next_active

    results = []
    for s in seqs:
        item = questions[s["qi"]]
        answers = item.get("gold_answers") or [item["answer"]]
        reward, info = -1e9, None
        for a in answers:
            r, inf = compute_reward_m5(
                s["turn_texts"], a, item.get("gold_titles", set()),
                s["retrieved_titles"], search_chars=s["search_chars"],
                max_search=MAX_SEARCH, max_turns=MAX_TURNS,
            )
            if r > reward:
                reward, info = r, inf
        results.append({
            "qi": s["qi"], "question": item["question"],
            "gold": item["answer"], "messages": s["messages"],
            "reward": reward, **info,
        })
    return results, llm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/m3_run2/iter0_merged")
    ap.add_argument("--data", default="eval")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--G", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--idx_dir", default="index")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--gpu_util", type=float, default=0.7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    questions = load_eval_questions(args.data, args.n)
    results, _ = rollout(args.model, questions, G=args.G,
                         temperature=args.temperature, idx_dir=args.idx_dir,
                         max_new_tokens=args.max_new_tokens,
                         gpu_util=args.gpu_util)
    n = len(results)
    em_like = sum(1 for r in results if r["r_answer"] == 1.0) / n
    avg_f1 = sum(r["r_answer"] for r in results) / n
    avg_search = sum(r["n_search"] for r in results) / n
    fmt = sum(r["r_format"] for r in results) / n
    k_spec = sum(r["k_specified_pct"] for r in results) / n
    avg_k = (sum(r["avg_k"] for r in results if r["n_search"])
             / max(1, sum(1 for r in results if r["n_search"])))
    chars = sum(r["search_chars"] for r in results) / n
    print(json.dumps({"model": args.model, "data": args.data, "n_q": n,
                      "avg_answer_f1": round(avg_f1, 4),
                      "pct_perfect_answer": round(em_like * 100, 2),
                      "format_ok_pct": round(fmt * 100, 2),
                      "avg_n_search": round(avg_search, 2),
                      "k_specified_pct": round(k_spec * 100, 2),
                      "avg_k_when_searched": round(avg_k, 2),
                      "avg_search_chars": round(chars, 1)},
                     ensure_ascii=False, indent=1))
    out = args.out or f"logs/m5_rollout_{args.data}_{n}.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"saved {len(results)} trajectories to {out}")
