"""substr 锚定质量抽检：打印 (教师译名 -> 规范标题) 及长度比，人工判断误配率"""
import json

from zh_anchor import TitleAnchor

ta = TitleAnchor()
rows = [json.loads(l) for l in open("data/zh_translated_train.jsonl")]
shown, checked = 0, 0
ratios = []
for r in rows:
    if not r["parse_ok"]:
        continue
    for tz in r["titles_zh"]:
        canon, method = ta.anchor(tz)
        if method == "substr":
            from m1_baseline_c import normalize_answer
            q = normalize_answer(tz)
            cq = normalize_answer(canon) if canon else ""
            ratios.append(len(cq) / max(len(q), 1))
            if shown < 25:
                print(f"  {tz!r} -> {canon!r}  ratio={len(cq)/max(len(q),1):.2f}")
                shown += 1
        checked += 1
    if checked > 1500:
        break
import numpy as np
r = np.array(ratios)
print(f"\nsubstr cases={len(r)}  ratio p10={np.percentile(r,10):.2f} "
      f"p50={np.percentile(r,50):.2f}  frac<0.6={float((r<0.6).mean()):.2f}")
