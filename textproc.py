"""共享分词（英文 + 中文），供建索引与检索端使用同一实现

英文：小写 + 仅保留 a-z0-9 后按空白切词（与 wiki-18 索引口径一致）
中文：连续 CJK 段切字 bigram（中文 IR 廉价而有效的经典做法）
纯英文文本的输出与历史版本逐 token 一致（向后兼容，英文索引可复用）
"""
import re

_TABLE = str.maketrans(
    {chr(c): " " for c in range(256) if not ("a" <= chr(c) <= "z" or "0" <= chr(c) <= "9")}
)

_CJK_RUN = re.compile(r"([一-鿿]+)")


def tokenize(text: str):
    out = []
    for seg in _CJK_RUN.split(text.lower()):
        if not seg:
            continue
        if "一" <= seg[0] <= "鿿":
            if len(seg) == 1:
                out.append(seg)
            else:
                out.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            out.extend(seg.translate(_TABLE).split())
    return out


def has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)
