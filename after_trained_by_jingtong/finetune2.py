### 这个py文件主要是LORA的微调训练，还有IA3模式，微调训练的参数更少

from dataclasses import dataclass, field
import datasets
from transformers import Trainer, HfArgumentParser, AutoTokenizer, AutoModel, TrainingArguments
from transformers import PreTrainedTokenizerBase
import torch
from transformers.trainer import TRAINING_ARGS_NAME
from peft import get_peft_model, LoraConfig, TaskType
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn


train_num = 500
model_path="/root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/bf0f5cfb575eebebf9b655c5861177acfee03f16/"

@dataclass
class FinetuneArguments:
    # dataset_path: str = field(default="/content/ChatGLM-Tuning/data/alpaca_data.json")
    # dataset_path: str = field(default="/content/ChatGLM-Tuning/data/alpaca_data.jsonl")
    dataset_path: str = field(default="/content/ChatGLM-Tuning/data/alpaca/data-00000-of-00001.arrow")
    model_path: str = field(default=model_path)  # 设置默认的模型路径
    lora_rank: int = field(default=8)

@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list) -> dict:
        print("features: ", features)
        len_ids = [len(feature["input_ids"]) for feature in features]
        longest = max(len_ids)
        input_ids = []
        labels_list = []
        for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            ids = feature["input_ids"]
            seq_len = feature["seq_len"]
            labels = (
                [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
            )
            ids = ids + [self.tokenizer.pad_token_id] * (longest - ids_l)
            _ids = torch.LongTensor(ids)
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

class ModifiedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 定义损失函数
        self.loss_fn = nn.CrossEntropyLoss()  # 可以根据需要调整
        self.vocab_size = 130528

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  #获取模型的输出
        # 这里可能需要根据你的模型输出调整
        # print("Logits shape:", logits.shape)  # 打印 logits 的形状
        # print("Labels shape:", labels.shape)  # 打印 labels 的形状
        # Logits shape: torch.Size([3, 81, 130528])
        # 3:batch size; 81 sequence length; 130528 vocabsize, 元素总数3*81*130528

        # Labels shape: torch.Size([3, 81])
        # logits转换为二维形状，lobels转换为一维形状
        loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss is NaN or Inf!")
        else:
            print(f"Computed loss: {loss.item()}")  # 打印损失值
        return (loss, outputs) if return_outputs else loss

    # def save_model(self, output_dir=None, _internal_call=False):
    #     os.makedirs(output_dir, exist_ok=True)
    #     torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    #     saved_params = {
    #         k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
    #     }
    #     torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


@dataclass
class CustomTrainingArguments(TrainingArguments):
    # 解决问题 IndexError: Invalid key: 16 is out of bounds for size 0
    remove_unused_columns: bool = field(default=False)  # 默认值设置为 False
    fp16: bool = field(default=True)
    learning_rate: float = field(default=1e-4)
    max_steps: int = field(default=3860)
    # per_device_train_batch_size: int = field(default=6)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=2)
    logging_steps: int = field(default=50)

def main(output_dir):
    writer = SummaryWriter()

    # Parse arguments using HfArgumentParser
    parser = HfArgumentParser((FinetuneArguments, CustomTrainingArguments))
    finetune_args, training_args = parser.parse_args_into_dataclasses()
    # print(training_args.fp16)
    # print(training_args.gradient_accumulation_steps)
    # print(training_args.per_device_train_batch_size)
    # print(training_args.learning_rate)
    # print(training_args.max_steps)
    # print(training_args.logging_steps)
    # print(training_args.remove_unused_columns)
    # print(training_args.seed)
    # print(training_args.data_seed)
    # print(training_args.group_by_length)


    # training_args = TrainingArguments(
    #     "output",
    #     fp16=True,  False
    #     gradient_accumulation_steps=1,  1
    #     per_device_train_batch_size=1,  8
    #     learning_rate=1e-4,  5e-05
    #     max_steps=1500,  -1
    #     logging_steps=50,  500
    #     remove_unused_columns=False,  False
    #     seed=0,
    #     data_seed=0,  None
    #     group_by_length=False,   False
    # )
    # False
    # 1
    # 8
    # 5e-05
    # -1
    # 500
    # False
    # None
    # False

    # Ensure output_dir is set correctly
    training_args.output_dir = output_dir  # 或根据需要修改为其他路径

    # Initialize model
    model = AutoModel.from_pretrained(
        "THUDM/chatglm-6b",
        load_in_8bit=True,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, trust_remote_code=True)

    # Configure model
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False  # Silence the warnings

    # Setup PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, #这个类型有很多，需要查一下
        inference_mode=False, #是否是推理模式
        r=finetune_args.lora_rank, #降秩矩阵的尺寸，这个参数会影响训练的参数量
        lora_alpha=32, #Lora的缩放系数，不影响参数量
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    # Load dataset
    # dataset = datasets.load_dataset('json', data_files=finetune_args.dataset_path)
    dataset = datasets.load_dataset('arrow', data_files=finetune_args.dataset_path)
    train_dataset = dataset['train']
    # print(f"\n{len(train_dataset)=}\n")
    # print("---",dataset)

    num_rows = len(train_dataset)
    print(f"\nNumber of rows in train dataset: {num_rows}\n")

    # Split the dataset (80% for training and eval, 20% for validation)
    # Device the train datater into three parts: 36600 train, 500 eval in train, 10000 eval after train,
    train_size = int(0.8 * num_rows)  # 80% as training set
    validation_size = num_rows - train_size

    # train_split = train_dataset.select(range(train_size))  # Select first 80%
    train_split = train_dataset.select(range(11, train_size-5000))  # Select first 80%-5000 for train
    # print("train_split: ", train_split)
    validation_split = train_dataset.select(range(train_size-4999, train_size))  # Select 5000 作为训练时评估
    # print("validation_split: ", validation_split)
    final_validation = train_dataset.select(range(train_size, num_rows)) #余下的百分之20%用来评估

    # Define metrics for evaluation (optional, depending on task)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        # print("logits: ", logits)
        # print("labels: ", labels)
        # 确保 logits 是一个 PyTorch 张量
        if not isinstance(logits, torch.Tensor):
            print("Change logits to tensor")
            logits = torch.tensor(logits)

        predictions = logits.argmax(dim=-1)
        # Assume task is text generation, customize metrics accordingly
        accuracy = (predictions == labels).float().mean().item()
        return {"accuracy": accuracy}

    # Start training
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_split,
        eval_dataset=validation_split,  # Add validation dataset here
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=DataCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,  # Add metrics if you want
    )
    trainer.train()

    # Save model
    print("output_dir: ", training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # 清空cuda缓存
    torch.cuda.empty_cache()

    # Evaluate the model on the validation dataset after training
    with torch.no_grad():
        eval_results = trainer.evaluate(eval_dataset=final_validation)
        print(f"Evaluation results: {eval_results}")

        for key, value in eval_results.items():
            writer.add_scalar(f"eval/{key}", value, trainer.state.global_step)  # 使用 global_step 表示训练步数

    writer.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the model")
    args = parser.parse_args()

    main(args.output_dir)
    # dataset = datasets.load_dataset('json', data_files="/Users/melissa/github/ChatGLM-Tuning/data/alpaca_data.json")
    # print(f"\n{len(dataset)=}\n")



