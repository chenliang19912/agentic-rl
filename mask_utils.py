"""对话轨迹编码与 assistant-only 损失 mask（M2/M3 共用）

实测结论（transformers 5.14 + Qwen3 模板）：
- 官方 return_assistant_tokens_mask：模板无 {% generation %}，返回全 0 → 不可用
- 前缀长度差分：模板按上下文给 assistant 注入 think 块，前缀渲染与全量渲染不一致 → 不可用
- 字符级锚点 + offset_mapping：tok(text) 与模板 tokenize 的 ids 完全一致 → 精确可靠
"""
import torch


def encode_chat_assistant_only(tok, messages, max_len=3072):
    """返回 (input_ids, attention_mask, labels)；labels 仅在 assistant 内容上有效。"""
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"][:max_len]
    offsets = enc["offset_mapping"][:max_len]
    labels = [-100] * len(ids)
    pos = 0
    for m in messages:
        if m["role"] != "assistant":
            continue
        c = m["content"]
        if not c:
            continue
        start = text.find(c, pos)
        if start < 0:
            continue
        end = start + len(c)
        pos = end
        for j, (s, e) in enumerate(offsets):
            if e == 0:
                continue
            if s < end and e > start:
                labels[j] = ids[j]
    return ids, [1] * len(ids), labels


def encode_chat_assistant_turns(tok, messages, max_len=3072):
    """轮边界版编码（WS-2 turn 级优势用）：返回 (input_ids, attention_mask, labels, spans)。

    与 encode_chat_assistant_only 完全同源（字符锚点 + offset_mapping，已实测可靠），
    只增加每条 assistant 消息的 token 跨度记录；labels 语义与原函数逐位一致。
    spans = [(start_tok, end_tok, assistant序号j)]，end 独占；j 按 messages 中
    assistant 消息出现顺序编号（与 c4_gigpo.turn_anchors 的 j 对齐——空内容/未命中
    锚点的轮保留编号但无 span，其优势不进 loss，由调用方计入 n_turns_truncated）。
    max_len 截断时 span 随 ids/offsets 同步裁剪（超出部分自然消失）。
    """
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"][:max_len]
    offsets = enc["offset_mapping"][:max_len]
    labels = [-100] * len(ids)
    spans = []
    pos = 0
    a_idx = 0
    for m in messages:
        if m["role"] != "assistant":
            continue
        c = m["content"]
        cur = a_idx
        a_idx += 1
        if not c:
            continue
        start = text.find(c, pos)
        if start < 0:
            continue
        end = start + len(c)
        pos = end
        tok_start, tok_end = None, None
        for j, (s, e) in enumerate(offsets):
            if e == 0:
                continue
            if s < end and e > start:
                labels[j] = ids[j]
                if tok_start is None:
                    tok_start = j
                tok_end = j + 1
        if tok_start is not None:
            spans.append((tok_start, tok_end, cur))
    return ids, [1] * len(ids), labels, spans


def to_tensors(sample):
    return (torch.tensor(sample["input_ids"], dtype=torch.long),
            torch.tensor(sample["attention_mask"], dtype=torch.long),
            torch.tensor(sample["labels"], dtype=torch.long))
