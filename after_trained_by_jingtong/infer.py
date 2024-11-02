#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Li jinjing <lijinjing@gmail.com>
import sys
import os
import json
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer
print(os.getcwd())
sys.path.append(os.path.abspath(os.getcwd()+"/ChatGLM-Tuning"))


from cover_alpaca2jsonl import format_example


#model = PeftModel.from_pretrained(model, model_name)

model_name="/content/ChatGLM-Tuning/model"

# 加载基础模型
#base_model = AutoModel.from_pretrained(model_name)

# 使用 LoRA 的 PeftModel 来加载 LoRA 权重
config = PeftConfig.from_pretrained(model_name)
print(config)
# 重启初始化原模型
base_model = AutoModel.from_pretrained(config.base_model_name_or_path).half()  # 使用 FP16
print(base_model)
# 插入保存的lora层
model = PeftModel.from_pretrained(base_model, model_name).half()  # 使用 FP16
print(model)

## 除了插入还可以将Lora层合并到原始模型中，效果不会改变，但会丢失lora层的参数
# model_merge = model.merge_and_unload().....

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
model.eval()  # 切换到推理模式

instructions = json.load(open("data/alpaca_data.json"))


with torch.no_grad():
    for idx, item in enumerate(instructions[:5]):
        feature = format_example(item)
        input_text = feature["context"]
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        print(f"Start generating for example {idx + 1}...")
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            temperature=0
        )
        answer = tokenizer.decode(out[0])
        print(answer)
        item['infer_answer'] = answer
        print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
