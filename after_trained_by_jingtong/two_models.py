#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Li jinjing <lijinjing@gmail.com>
ChatGLMForConditionalGeneration(
  (transformer): ChatGLMModel(
    (word_embeddings): Embedding(130528, 4096)
    (layers): ModuleList(
      (0-27): 28 x GLMBlock(
        (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (attention): SelfAttention(
          (rotary_emb): RotaryEmbedding()
          (query_key_value): Linear(in_features=4096, out_features=12288, bias=True)
          (dense): Linear(in_features=4096, out_features=4096, bias=True)
        )
        (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (mlp): GLU(
          (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)
          (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)
        )
      )
    )
    (final_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4096, out_features=130528, bias=False)
)
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): ChatGLMForConditionalGeneration(
      (transformer): ChatGLMModel(
        (word_embeddings): Embedding(130528, 4096)
        (layers): ModuleList(
          (0-27): 28 x GLMBlock(
            (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (attention): SelfAttention(
              (rotary_emb): RotaryEmbedding()
              (query_key_value): lora.Linear(
                (base_layer): Linear(in_features=4096, out_features=12288, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=4096, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=12288, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (dense): Linear(in_features=4096, out_features=4096, bias=True)
            )
            (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (mlp): GLU(
              (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)
              (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)
            )
          )
        )
        (final_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=4096, out_features=130528, bias=False)
    )
  )
)
