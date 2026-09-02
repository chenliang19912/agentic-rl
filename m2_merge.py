"""合并 M2 LoRA adapter 到基座，输出完整模型目录供 M3 rollout 使用"""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/m2_sft_lora"
OUT = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/m2_sft_merged"
# C4 用 Instruct-2507 基座，通过第 3 参传入；默认值保持检索 Agent 项目行为不变
BASE = sys.argv[3] if len(sys.argv) > 3 else "/models/Qwen3-4B"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="cpu",
)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print(f"merged model saved to {OUT}")
