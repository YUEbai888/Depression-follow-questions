# -*- coding: utf-8 -*-

"""对t5进行微调"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import json
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
class T5Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.inputs = data['x']
            self.targets = data['y']
            assert len(self.inputs) == len(self.targets), "输入和输出长度不匹配"
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        # 对输入和输出进行编码
        inputs = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        targets = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        labels = targets["input_ids"].squeeze()
        # 让 loss 忽略 <pad>
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }

def main():
    # 初始化tokenizer和模型
    model_name = 't5'  # 使用t5-base模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # model.gradient_checkpointing_enable() # 一会儿看看是否使用
    # 加载数据集
    train_dataset = T5Dataset('./dataset/processed_t5_datasets/train/t5_dataset.json', tokenizer)
    
    # 设置训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir           = "./t5_finetuned",
        num_train_epochs     = 2,
        per_device_train_batch_size = 8,
        learning_rate        = 2e-4,
        warmup_steps         = 500,
        weight_decay         = 0.01,
        logging_dir          = "./logs",
        logging_steps        = 100,
        save_strategy        = "steps",
        save_steps           = 500,
        save_total_limit     = 1,  # ✅ 只保留最新 checkpoint，自动删除旧的
        fp16 = torch.cuda.is_available(),
        gradient_accumulation_steps = 4,
    )

    
    # 初始化数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    # 初始化训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model('./t5_final')
    
if __name__ == "__main__":
    main()


