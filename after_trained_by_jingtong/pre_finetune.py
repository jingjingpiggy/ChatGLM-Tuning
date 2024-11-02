# from transformers import AutoTokenizer,AutoModel, AutoModelForCausalLM

# !git clone https://github.com/mymusise/ChatGLM-Tuning.git
# %cd  ChatGLM-Tuning
# !pip install transformers==4.33.1 datasets accelerate  bitsandbytes  peft protobuf==3.20.*
# transformers 4.33.1 在A100上是ok的

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

## CONVERT JSON TO JSONL
# !python cover_alpaca2jsonl.py \
#     --data_path data/alpaca_data.json \
#     --save_path data/alpaca_data.jsonl


# !mv /content/finetune2.py /content/ChatGLM-Tuning
# !ls -l /content/ChatGLM-Tuning/model



### DOWNLOAD MODEL FROM HUGGING FACE
# # # 加载预训练模型的分词器
# model = AutoModel.from_pretrained("THUDM/chatglm-6b")
# print(model)

# # 使用 Huggingface 的模型名称加载 ChatGLM 分词器和模型
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# # 定义提示
# prompt = "请给我讲个玫瑰的爱情故事?"
# # 分词
# inputs = tokenizer(prompt, return_tensors="pt")
# # 生成
# outputs = model.generate(inputs["input_ids"], max_new_tokens=200)
# # 解码输出
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)

## TOKENIZE JSON DATA
# !python tokenize_dataset_rows.py --jsonl_path data/alpaca_data.jsonl \
# --save_path data/alpaca --max_seq_length 200 --skip_overlength False \
# --chatglm_path /root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/bf0f5cfb575eebebf9b655c5861177acfee03f16/  \
# --version v1

### RUN test_before_finetune.py
### RUN FUNETUNE
# !python3 finetune2.py --output_dir /content/ChatGLM-Tuning/model

### RUN infer.py

## 模型形状
# ChatGLMForConditionalGeneration(
#   (transformer): ChatGLMModel(
#     (word_embeddings): Embedding(130528, 4096)
#     (layers): ModuleList(
#       (0-27): 28 x GLMBlock(
#         (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
#         (attention): SelfAttention(
#           (rotary_emb): RotaryEmbedding()
#           (query_key_value): Linear(in_features=4096, out_features=12288, bias=True)
#           (dense): Linear(in_features=4096, out_features=4096, bias=True)
#         )
#         (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
#         (mlp): GLU(
#           (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)
#           (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)
#         )
#       )
#     )
#     (final_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
#   )
#   (lm_head): Linear(in_features=4096, out_features=130528, bias=False)
# )

# PeftModelForCausalLM(
#   (base_model): LoraModel(
#     (model): ChatGLMForConditionalGeneration(
#       ## 模型部分-ChatGLM
#       (transformer): ChatGLMModel(
#         # Embedding层
#         (word_embeddings): Embedding(130528, 4096) #词嵌入层的维度为 (130528, 4096)，表示模型的词汇表大小为 130528，嵌入的每个词向量维度是 4096。
#         (layers): ModuleList(
#           #  GLMBlock：28层的Transformer Block内部包含自注意力机制和MLP模块
#           (0-27): 28 x GLMBlock(
#             (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
#             (attention): SelfAttention( #自注意力机制
#               (rotary_emb): RotaryEmbedding() #通过旋转位置编码（RotaryEmbedding）以及 query_key_value（QKV）来计算注意力分数。
#               (query_key_value): lora.Linear( #使用了 LoRA 技术，在这个地方通过降维（lora_A）和升维（lora_B）来降低计算成本。
#                 (base_layer): Linear(in_features=4096, out_features=12288, bias=True)
#                 (lora_dropout): ModuleDict(
#                   (default): Dropout(p=0.1, inplace=False) #防止过拟合
#                 )
#                 (lora_A): ModuleDict(
#                   (default): Linear(in_features=4096, out_features=8, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (default): Linear(in_features=8, out_features=12288, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#                 (lora_magnitude_vector): ModuleDict()
#               )
#               (dense): Linear(in_features=4096, out_features=4096, bias=True)       #线性层，用于调整自注意力的输出维度
#             )
#             (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True) #注意力之后的标准化层
#             (mlp): GLU(                                                                #是多层感知机（MLP）的变种
#               (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True) #将 4096 维输入映射到 16384 维（增加表示能力）
#               (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True) #将 16384 维再映射回 4096 维
#             )
#           )
#         )
#         # 最终的标准化层，在 Transformer 模型中，标准化层常用于控制数据分布，帮助模型收敛。
#         (final_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
#       )
#       # 线性层lm_head是模型的输出层，它将来自上面的特征映射到词汇表大小的维度（in_features=4096, out_features=130528），即将特征转化为每个词的概率分布。
#       (lm_head): Linear(in_features=4096, out_features=130528, bias=False)
#     )
#   )
# )
