#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Li jinjing <lijinjing@gmail.com>
###finetune前的推理
import sys
import os
# print(os.getcwd())
sys.path.append(os.path.abspath(os.getcwd()+"/ChatGLM-Tuning"))

from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig, AutoModelForCausalLM
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from cover_alpaca2jsonl import format_example
import json


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)

model_name="/Users/melissa/Downloads/model/adapter_model.safetensors"

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True, device_map='auto')
model.supports_gradient_checkpointing = True
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.lm_head = CastOutputToFloat(model.lm_head)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

instructions = json.load(open("/content/ChatGLM-Tuning/data/alpaca_data.json"))


with torch.no_grad():
    for idx, item in enumerate(instructions[:5]):
        feature = format_example(item)
        input_text = feature["context"]
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            temperature=0
        )
        answer = tokenizer.decode(out[0])
        print(answer)
        item['infer_answer'] = answer
        print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
