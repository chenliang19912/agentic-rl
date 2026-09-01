"""M0 验证脚本：多轮 <search> → 检索 → 续生成 → <answer> rollout

用法（宿主机）：
docker run --rm --gpus '"device=0"' \
  -v /mnt/nas3/shared/model:/models:ro \
  -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work \
  -w /work -e HF_HUB_OFFLINE=1 \
  1cat-vllm:v100-1.3.0 python m0_rollout.py
"""
import json
import math
import re
from collections import Counter

from vllm import LLM, SamplingParams

MODEL = "/models/Qwen3-4B"
MAX_TURNS = 3
TOP_K = 2

# ---------- M0 迷你语料（两跳链条 + 干扰项）----------
CORPUS = [
    {"id": "d1", "text": "The film Inception was directed by Christopher Nolan and released in 2010."},
    {"id": "d2", "text": "Christopher Nolan is a British filmmaker who was born in London in 1970."},
    {"id": "d3", "text": "London is the capital city of England and the United Kingdom."},
    {"id": "d4", "text": "The film Titanic was directed by James Cameron and released in 1997."},
    {"id": "d5", "text": "Paris is the capital city of France, known for the Eiffel Tower."},
    {"id": "d6", "text": "The Dark Knight is a 2008 superhero film directed by Christopher Nolan."},
]

QUESTION = "In which city was the director of the film Inception born?"
GOLD = "london"

# ---------- 极简 BM25 ----------
_token = lambda s: re.findall(r"[a-z0-9]+", s.lower())
DOCS_TOK = [_token(d["text"]) for d in CORPUS]
DF = Counter(t for doc in DOCS_TOK for t in set(doc))
N, AVGDL = len(CORPUS), sum(len(d) for d in DOCS_TOK) / len(CORPUS)


def bm25(query: str, k1: float = 1.5, b: float = 0.75):
    qtok, scores = _token(query), []
    for i, dtok in enumerate(DOCS_TOK):
        tf = Counter(dtok)
        s = sum(
            math.log(1 + (N - DF[t] + 0.5) / (DF[t] + 0.5)) * tf[t] * (k1 + 1)
            / (tf[t] + k1 * (1 - b + b * len(dtok) / AVGDL))
            for t in qtok if t in tf
        )
        scores.append((s, i))
    scores.sort(reverse=True)
    return [CORPUS[i] for s, i in scores[:TOP_K] if s > 0]


# ---------- rollout 循环 ----------
SYSTEM = (
    "You are a question-answering agent. Answer the question using search results.\n"
    "To search, output exactly: <search>your query</search>\n"
    "When you can answer, output exactly: <answer>your answer</answer>\n"
    "Search at most 3 times. Answer must be concise."
)


def main():
    llm = LLM(
        model=MODEL, dtype="float16", gpu_memory_utilization=0.5,
        max_model_len=4096, enforce_eager=True,
    )
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=256, temperature=0.0, stop=["</search>", "</answer>"])

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {QUESTION}"},
    ]
    trajectory = []
    final_answer = None

    for turn in range(1, MAX_TURNS + 2):
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            extra_kwargs={"enable_thinking": False},
        )
        raw = llm.generate([prompt], sp)[0].outputs[0].text
        if "<search>" in raw:
            query = raw.split("<search>", 1)[1].strip()
            hits = bm25(query)
            results = "\n".join(f"- {d['text']}" for d in hits) or "(no results)"
            trajectory.append({"turn": turn, "action": "search", "query": query, "hits": [d["id"] for d in hits]})
            messages.append({"role": "assistant", "content": f"<search>{query}</search>"})
            messages.append({"role": "user", "content": f"Search results for '{query}':\n{results}"})
        else:
            final_answer = raw.strip()
            trajectory.append({"turn": turn, "action": "answer", "text": final_answer})
            break

    ok = final_answer is not None and GOLD in final_answer.lower()
    n_search = sum(1 for t in trajectory if t["action"] == "search")
    print("\n===== M0 ROLLOUT RESULT =====")
    print(json.dumps(trajectory, ensure_ascii=False, indent=1))
    print(f"question   : {QUESTION}")
    print(f"final      : {final_answer}")
    print(f"gold       : {GOLD}")
    print(f"n_search   : {n_search}")
    print(f"M0_MULTI_TURN_ROLLOUT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
