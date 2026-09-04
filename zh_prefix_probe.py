"""诊断脚本：截断前缀续写实验（v4 作答率 0% 的机理定位）

问题：v4（教师轨迹式）训练数据答案轮占比 26.2%（高于英文 v1 的 16.0%），
但温度 1.0 探查 120 条轨迹全部搜满不答。两个竞争假设：
H1「答案分支没学会」：任何前缀下都不产 <answer>；
H2「分布迁移」：训练分布前缀下会答，评测分布前缀下不答。

实验：把轨迹截断到「搜索结果刚出现」处让模型续写下一轮：
- A 组：训练集 one_hop 前缀（问题+搜索+结果）20 条
- B 组：训练集 two_hop 前缀（两次搜索+结果都出现）20 条
- C 组：评测探查轨迹 probe_zh_v4.jsonl 的第一跳后前缀 20 条
统计各组续写中 <answer> / <search> 出现率。
"""
import json

from vllm import LLM, SamplingParams

MODEL = "checkpoints/m2_sft_zh_merged_v5"


def prefix_upto(messages, upto_role_count):
    """截到第 upto_role_count 条消息（含）"""
    return messages[:upto_role_count]


def main():
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.6,
              max_model_len=8192, enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=300, temperature=0.0)

    train = [json.loads(l) for l in open("data/sft_train_zh_v1style.jsonl")]
    one_hop = [r for r in train if r["meta"]["type"] == "one_hop"][:20]
    two_hop = [r for r in train if r["meta"]["type"] == "two_hop"][:20]
    probe = json.load(open("logs/probe_zh_v4.jsonl"))[:20]

    groups = {
        "A_train_one_hop": [prefix_upto(r["messages"], 4) for r in one_hop],
        # sys, q, search1, result1, search2, result2 → 6 条后模型该答了
        "B_train_two_hop": [prefix_upto(r["messages"], 6) for r in two_hop],
        # 探查轨迹：sys, q, search1, result1 → 4 条
        "C_eval_turn1": [prefix_upto(r["messages"], 4) for r in probe],
    }

    for name, prefixes in groups.items():
        prompts = [tok.apply_chat_template(p, tokenize=False,
                                           add_generation_prompt=True)
                   for p in prefixes]
        outs = llm.generate(prompts, sp)
        n_ans = sum("<answer>" in o.outputs[0].text for o in outs)
        n_srch = sum("<search>" in o.outputs[0].text for o in outs)
        print(f"{name}: n={len(outs)} answer={n_ans} search={n_srch}")
        for o in outs[:2]:
            print("   e.g.:", o.outputs[0].text[:220].replace("\n", " | "))


if __name__ == "__main__":
    main()
