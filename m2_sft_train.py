"""M2 SFT 冷启动训练：Qwen3-4B + LoRA，学习 <search>/<answer> 轨迹格式

数据：data/sft_train.jsonl（m2_sft_data.py 生成，messages 对话格式）
损失仅计 assistant 内容：自行用 mask_utils（字符锚点 + offset_mapping，已实测）
预 tokenize 成 input_ids/labels。训练用原生 transformers Trainer + PEFT，
不依赖 trl 的新版 API（本机 transformers 5.14 过新，求稳）。

V100 适配：fp16、gradient checkpointing。

用法（单卡）：
docker run --rm --gpus '"device=2"' \
  -v /mnt/nas3/shared/model:/models:ro \
  -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
  -e HF_HUB_OFFLINE=1 \
  1cat-vllm:v100-1.3.0 python -u m2_sft_train.py
"""
import json
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

from mask_utils import encode_chat_assistant_only

MODEL = "/models/Qwen3-4B"
# 第 6 个参数：基座路径。续训场景传已合并的 checkpoint（如 m2_sft_zh_merged_v4），
# 在其上叠新 LoRA 继续加曝光（中文线答案分支 1 epoch 翻不过来的对策）
if len(sys.argv) > 6:
    MODEL = sys.argv[6]
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/sft_train.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/m2_sft_lora"
# 轨迹长度分布：英文单跳 ~850、两跳 ~1600，2048 足够；
# 中文多跳轨迹更长（字符 p95≈3150），2048 会截掉尾部 <answer> 轮 → 用 4096
MAX_LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
BS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ACC = max(1, 8 // BS)   # 有效 batch 维持 8
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 3

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

samples, skipped = [], 0
for line in open(DATA):
    r = json.loads(line)
    ids, attn, labels = encode_chat_assistant_only(tok, r["messages"], MAX_LEN)
    if sum(1 for l in labels if l != -100) < 5:
        skipped += 1
        continue
    samples.append({"input_ids": ids, "attention_mask": attn, "labels": labels})
print(f"tokenized {len(samples)} trajectories (skipped {skipped})", flush=True)

# 长度分布 + 按长度排序（配合顺序采样近似长度分桶，减少 padding 浪费）
lens = sorted(len(s["input_ids"]) for s in samples)
print(f"len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}", flush=True)
samples.sort(key=lambda s: len(s["input_ids"]))

ds = Dataset.from_list(samples)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
model.config.use_cache = False
peft_cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
model = get_peft_model(model, peft_cfg)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=EPOCHS,     # 冷启动：终止行为（<answer> 收尾）需要充分曝光
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=ACC,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    seed=7,
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tok, padding="longest"),
    processing_class=tok,
)
trainer.train()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"saved LoRA adapter to {OUT}")
