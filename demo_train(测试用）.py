# -*- coding: utf-8 -*-
"""对 T5 进行全量微调"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import nltk
nltk.data.path.append(r'D:/code/JIEDAN/Follow-up_Question/pkg/nltk_data')

# 检查并下载punkt和punkt_tab（如果没找到的话）
for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

# 测试是否能正常使用sent_tokenize
from nltk.tokenize import sent_tokenize
class T5Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, sample_ratio=0.02):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 检查数据集文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件 {data_path} 不存在")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            total_samples = len(data['x'])
            sample_size = int(total_samples * sample_ratio)
            indices = np.random.choice(total_samples, sample_size, replace=False)
            self.inputs = [data['x'][i] for i in indices]
            self.targets = [data['y'][i] for i in indices]
            assert len(self.inputs) == len(self.targets), "输入和输出长度不匹配"
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        inputs = self.tokenizer(
            input_text, max_length=self.max_length, padding='max_length', 
            truncation=True, return_tensors='pt'
        )
        targets = self.tokenizer(
            target_text, max_length=self.max_length, padding='max_length', 
            truncation=True, return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def compute_metrics(eval_pred, tokenizer, model, eval_dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()
    preds, labels = eval_pred
    
    # 确保 labels 是 numpy 数组并替换填充标记
    if isinstance(labels, tuple):
        labels = labels[0]
    labels = np.asarray(labels)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # 批量解码标签
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 使用 model.generate 生成预测
    decoded_preds = []
    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # 生成预测
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        
        
        # 解码生成的结果
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_preds.extend(batch_preds)
    
    # 确保预测和标签数量匹配
    assert len(decoded_preds) == len(decoded_labels), "预测和标签数量不匹配"
    
    # 计算 ROUGE 分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(pred, label)
        rouge_scores.append({
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        })
    
    # 计算平均 ROUGE 分数
    avg_rouge = {
        'rouge1': np.mean([s['rouge1'] for s in rouge_scores]),
        'rouge2': np.mean([s['rouge2'] for s in rouge_scores]),
        'rougeL': np.mean([s['rougeL'] for s in rouge_scores])
    }
    
    # 计算 BLEU 分数
    bleu_scores = []
    smoothie = SmoothingFunction().method1
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_tokens = nltk.word_tokenize(pred)
        label_tokens = nltk.word_tokenize(label)
        score = sentence_bleu([label_tokens], pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(score)
    
    avg_bleu = np.mean(bleu_scores)
    
    # 计算不同阈值下的命中率
    thresholds = [0.4, 0.5, 0.7]
    hit_rates = {}
    for threshold in thresholds:
        rougeL_hit_rate = np.mean([1 if s['rougeL'] >= threshold else 0 for s in rouge_scores])
        bleu_hit_rate = np.mean([1 if score >= threshold else 0 for score in bleu_scores])
        hit_rates[f'rougeL_hit_rate_{threshold}'] = rougeL_hit_rate
        hit_rates[f'bleu_hit_rate_{threshold}'] = bleu_hit_rate
    
    return {
        'rouge1': avg_rouge['rouge1'],
        'rouge2': avg_rouge['rouge2'],
        'rougeL': avg_rouge['rougeL'],
        'bleu': avg_bleu,
        **hit_rates
    }

def show_model_outputs(model, tokenizer, val_dataset, num_samples=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()
    print("\n模型输出示例:")
    for i in tqdm(range(min(num_samples, len(val_dataset))), desc="Generating outputs"):
        input_text = val_dataset.inputs[i]
        target_text = val_dataset.targets[i]
        
        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
        
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n输入: {input_text}")
        print(f"目标输出: {target_text}")
        print(f"模型输出: {generated_text}")
        print("-" * 50)

def main():
    try:
        # 初始化 tokenizer 和模型
        model_name = 't5'  # 可改为 't5-small' 或 't5-large'，视资源情况
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # 加载并合并数据集
        train_data_path = 'dataset/processed_t5_datasets/train/t5_dataset.json'
        val_data_path = 'dataset/processed_t5_datasets/val/t5_dataset.json'
        
        # 读取训练集和验证集
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(val_data_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            
        # 合并数据集
        merged_data = {
            'x': train_data['x'] + val_data['x'],
            'y': train_data['y'] + val_data['y']
        }
        
        # 保存合并后的数据集
        merged_data_path = 'dataset/processed_t5_datasets/merged_dataset.json'
        os.makedirs(os.path.dirname(merged_data_path), exist_ok=True)
        with open(merged_data_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
        # 使用合并后的数据集
        full_dataset = T5Dataset(merged_data_path, tokenizer)
        
        # 设置训练参数
        training_args = Seq2SeqTrainingArguments(
            output_dir='./t5_finetuned',
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            fp16=torch.cuda.is_available(),
        )
        
        # 初始化数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
        
        # 创建验证集 DataLoader（使用部分数据作为验证集）
        val_size = int(len(full_dataset) * 0.1)  # 使用10%的数据作为验证集
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=data_collator
        )
        
        # 初始化训练器
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, model, val_dataloader),
        )
        
        # 开始训练
        print("开始训练...")
        trainer.train()
        
        # 保存模型
        trainer.save_model('./t5_finetuned_final')
        tokenizer.save_pretrained('./t5_finetuned_final')
        
        # 最终评估
        print("\n进行最终评估...")
        final_metrics = trainer.evaluate()
        print("\n最终评估结果:")
        print(f"ROUGE-1: {final_metrics['eval_rouge1']:.4f}")
        print(f"ROUGE-2: {final_metrics['eval_rouge2']:.4f}")
        print(f"ROUGE-L: {final_metrics['eval_rougeL']:.4f}")
        print(f"BLEU: {final_metrics['eval_bleu']:.4f}")
        print("\n不同阈值下的命中率:")
        for threshold in [0.4, 0.5, 0.7]:
            print(f"\n阈值 {threshold}:")
            print(f"ROUGE-L 命中率: {final_metrics[f'eval_rougeL_hit_rate_{threshold}']:.4f}")
            print(f"BLEU 命中率: {final_metrics[f'eval_bleu_hit_rate_{threshold}']:.4f}")
        
        # 展示模型输出
        show_model_outputs(model, tokenizer, val_dataset, num_samples=3)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()