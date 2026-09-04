"""中文 SFT 数据再平衡：提高 one_hop 样本占比

诊断背景：中文 775 条训练轨迹里 one_hop 仅 14%（英文线 ~34%），
且答案轮中位仅 12 token，多跳轨迹的搜索循环主导损失，
模型学会"抽实体再搜"却学不会"<answer> 收尾"（贪心作答率 0%）。

对策：把 one_hop 轨迹（搜索→即答，答案轮占比最高）重复 3 倍，
占比升到 ~33%，给终止行为更多曝光。
"""
import json

SRC = "data/sft_train_zh.jsonl"
OUT = "data/sft_train_zh_v2.jsonl"
REPEAT_ONE_HOP = 3

rows = [json.loads(l) for l in open(SRC)]
out = []
for r in rows:
    out.append(r)
    if r["meta"]["type"] == "one_hop":
        for _ in range(REPEAT_ONE_HOP - 1):
            out.append(r)

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

n_one = sum(1 for r in out if r["meta"]["type"] == "one_hop")
print(f"total {len(rows)} -> {len(out)} samples; "
      f"one_hop share {sum(1 for r in rows if r['meta']['type']=='one_hop')/len(rows)*100:.0f}%"
      f" -> {n_one/len(out)*100:.0f}%")
print(f"saved {OUT}")
