# -*- coding: utf-8 -*-
"""LoRAアダプタをbaseにマージしてbf16で保存(vLLMがMoEのLoRAを未対応なため)。
使い方: python merge_lora.py --base <hf> --lora ./ft/<x> --out /opt/llm/merged/<x>"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--lora", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    a.base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, a.lora)
model = model.merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
print(f"merged saved to {a.out}")
