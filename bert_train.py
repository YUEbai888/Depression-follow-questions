# -*- coding: utf-8 -*-
"""
训练bert模型 使用datasets.dataload 中的数据
数据集中x是文本表示 y是0or1
使用bert提取x中的特征 然后预测01
评估指标是 mcc
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import json
import numpy as np
from tqdm import tqdm
import os
import logging
import sys
import traceback
from datetime import datetime
import gc
from sklearn.metrics import matthews_corrcoef
import torch.nn.functional as F
import pandas as pd
import argparse

# 设置日志记录器的编码
logging.basicConfig(
    filename='training.log',
    encoding='utf-8',  # 明确指定 UTF-8 编码
    level=logging.INFO
)

# 默认参数
EPOCH = 8
BATCH = 64

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='BERT模型训练脚本')
    
    # 添加命令行参数
    parser.add_argument('--epochs', type=int, default=EPOCH,
                      help=f'训练轮数 (默认: {EPOCH})')
    parser.add_argument('--batch_size', type=int, default=BATCH,
                      help=f'批次大小 (默认: {BATCH})')
    parser.add_argument('--bert_path', type=str, default='bert',
                      help='BERT模型路径 (默认: bert)')
    parser.add_argument('--start_dataset', type=int, default=1,
                      help='起始数据集索引 (默认: 1)')
    parser.add_argument('--end_dataset', type=int, default=9,
                      help='结束数据集索引 (默认: 9)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='学习率 (默认: 2e-5)')
    parser.add_argument('--save_dir', type=str, default='bert_models',
                      help='模型保存目录 (默认: bert_models)')
    
    return parser.parse_args()

# 配置日志
def setup_logging(dataset_index):
    log_dir = f'logs/dataset_{dataset_index}'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# 清理GPU内存
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class PHQ9Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        try:
            # 加载数据
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.texts = data['x']
            self.labels = data['y']
        except Exception as e:
            logging.error(f"加载数据集失败: {str(e)}")
            raise
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        try:
            text = self.texts[idx]
            label = self.labels[idx]
            
            # 对文本进行编码，确保padding到max_length
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',  # 确保padding到max_length
                truncation=True,
                return_tensors='pt'
            )
            
            # 确保input_ids和attention_mask的长度都是max_length
            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()
            
            # 如果长度不足max_length，进行padding
            if len(input_ids) < self.max_length:
                padding_length = self.max_length - len(input_ids)
                input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logging.error(f"处理数据项 {idx} 时出错: {str(e)}")
            raise

def calculate_metrics(predictions, labels, threshold=0.5):
    """
    计算评估指标
    """
    try:
        # 将概率转换为二分类预测
        binary_predictions = (predictions >= threshold).astype(int)
        mcc = matthews_corrcoef(labels, binary_predictions)
        accuracy = (binary_predictions == labels).mean()
        return {
            'mcc': mcc,
            'accuracy': accuracy
        }
    except Exception as e:
        logging.error(f"计算指标时出错: {str(e)}")
        raise

def save_best_results(dataset_index, best_val_mccs, best_val_accs, results_file='best_results.csv'):
    """
    保存每个数据集在不同阈值下的最佳结果
    """
    try:
        # 准备数据
        data = {
            'dataset_index': dataset_index,
            'threshold_0.5_mcc': best_val_mccs[0.5],
            'threshold_0.5_acc': best_val_accs[0.5],
            'threshold_0.7_mcc': best_val_mccs[0.7],
            'threshold_0.7_acc': best_val_accs[0.7],
            'threshold_0.9_mcc': best_val_mccs[0.9],
            'threshold_0.9_acc': best_val_accs[0.9],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 如果文件存在，读取现有数据
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            # 如果该数据集的结果已存在，更新它
            if dataset_index in df['dataset_index'].values:
                df.loc[df['dataset_index'] == dataset_index] = pd.Series(data)
            else:
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        else:
            # 创建新的DataFrame
            df = pd.DataFrame([data])
        
        # 保存结果
        df.to_csv(results_file, index=False)
        logging.info(f'数据集 {dataset_index} 的最佳结果已保存到 {results_file}')
        
    except Exception as e:
        logging.error(f"保存最佳结果时出错: {str(e)}")
        logging.error(traceback.format_exc())

def train_model(model, train_loader, val_loader, device, dataset_index, num_epochs=EPOCH):
    logger = setup_logging(dataset_index)
    logger.info(f"开始训练数据集 {dataset_index} 的BERT模型")
    
    try:
        # 优化器
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        
        # 学习率调度器
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 为每个阈值创建最佳MCC和准确率记录
        best_val_mccs = {
            0.5: -1.0,
            0.7: -1.0,
            0.9: -1.0
        }
        best_val_accs = {
            0.5: 0.0,
            0.7: 0.0,
            0.9: 0.0
        }
        
        # 创建模型保存目录
        model_save_path = f'bert_models/dataset_{dataset_index}'
        os.makedirs(model_save_path, exist_ok=True)
        
        for epoch in range(num_epochs):
            logger.info(f'数据集 {dataset_index} - Epoch {epoch + 1}/{num_epochs}')
            
            # 训练阶段
            model.train()
            train_loss = 0
            all_train_probs = []
            all_train_labels = []
            
            try:
                for batch in tqdm(train_loader, desc=f'数据集 {dataset_index} - 训练中'):
                    try:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        model.zero_grad()
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        train_loss += loss.item()
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        
                        # 使用softmax获取概率
                        probs = F.softmax(outputs.logits, dim=1)[:, 1]  # 获取正类的概率
                        all_train_probs.extend(probs.cpu().detach().numpy())
                        all_train_labels.extend(labels.cpu().numpy())
                        
                        # 清理GPU内存
                        del outputs, loss, probs
                        clear_gpu_memory()
                        
                    except Exception as e:
                        logger.error(f"数据集 {dataset_index} - 训练批次处理出错: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue
                
                # 计算训练指标
                train_metrics = {}
                for threshold in [0.5, 0.7, 0.9]:
                    train_metrics[threshold] = calculate_metrics(np.array(all_train_probs), np.array(all_train_labels), threshold)
                avg_train_loss = train_loss / len(train_loader)
                
                # 验证阶段
                model.eval()
                val_loss = 0
                all_val_probs = []
                all_val_labels = []
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f'数据集 {dataset_index} - 验证中'):
                        try:
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch['labels'].to(device)
                            
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            
                            val_loss += outputs.loss.item()
                            
                            # 使用softmax获取概率
                            probs = F.softmax(outputs.logits, dim=1)[:, 1]  # 获取正类的概率
                            all_val_probs.extend(probs.cpu().numpy())
                            all_val_labels.extend(labels.cpu().numpy())
                            
                            # 清理GPU内存
                            del outputs, probs
                            clear_gpu_memory()
                            
                        except Exception as e:
                            logger.error(f"数据集 {dataset_index} - 验证批次处理出错: {str(e)}")
                            logger.error(traceback.format_exc())
                            continue
                
                # 计算验证指标
                val_metrics = {}
                for threshold in [0.5, 0.7, 0.9]:
                    val_metrics[threshold] = calculate_metrics(np.array(all_val_probs), np.array(all_val_labels), threshold)
                avg_val_loss = val_loss / len(val_loader)
                
                # 打印指标
                logger.info(f'数据集 {dataset_index} - Epoch {epoch + 1}/{num_epochs} 训练指标:')
                logger.info(f'  - 平均训练损失: {avg_train_loss:.4f}')
                for threshold in [0.5, 0.7, 0.9]:
                    logger.info(f'  - 阈值 {threshold} 训练MCC: {train_metrics[threshold]["mcc"]:.4f}')
                    logger.info(f'  - 阈值 {threshold} 训练准确率: {train_metrics[threshold]["accuracy"]:.4f}')
                
                logger.info(f'数据集 {dataset_index} - Epoch {epoch + 1}/{num_epochs} 验证指标:')
                logger.info(f'  - 平均验证损失: {avg_val_loss:.4f}')
                for threshold in [0.5, 0.7, 0.9]:
                    logger.info(f'  - 阈值 {threshold} 验证MCC: {val_metrics[threshold]["mcc"]:.4f}')
                    logger.info(f'  - 阈值 {threshold} 验证准确率: {val_metrics[threshold]["accuracy"]:.4f}')
                
                # 保存每个阈值的最佳模型
                for threshold in [0.5, 0.7, 0.9]:
                    if val_metrics[threshold]['mcc'] > best_val_mccs[threshold]:
                        best_val_mccs[threshold] = val_metrics[threshold]['mcc']
                        best_val_accs[threshold] = val_metrics[threshold]['accuracy']
                        best_model_path = os.path.join(model_save_path, f'best_model_threshold_{threshold}')
                        os.makedirs(best_model_path, exist_ok=True)
                        
                        # 保存模型配置和权重
                        model.save_pretrained(best_model_path)
                        
                        # 保存训练状态
                        training_state = {
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_mcc': best_val_mccs[threshold],
                            'val_acc': best_val_accs[threshold],
                            'threshold': threshold,
                            'dataset_index': dataset_index
                        }
                        torch.save(training_state, os.path.join(best_model_path, 'training_state.bin'))
                        
                        logger.info(f'数据集 {dataset_index} - 保存阈值 {threshold} 的最佳模型，MCC: {best_val_mccs[threshold]:.4f}，ACC: {best_val_accs[threshold]:.4f}，路径: {best_model_path}')
                
                # 在数据集训练完之后 需要记录一下在一个统一的文件写记录不同阈值下最佳mcc和acc
                if epoch == num_epochs - 1:  # 在最后一个epoch结束后保存结果
                    save_best_results(dataset_index, best_val_mccs, best_val_accs)
                
            except Exception as e:
                logger.error(f"数据集 {dataset_index} - Epoch {epoch + 1} 处理出错: {str(e)}")
                logger.error(traceback.format_exc())
                continue
            
            # 每个epoch结束后清理内存
            clear_gpu_memory()
            
    except Exception as e:
        logger.error(f"数据集 {dataset_index} - 训练过程出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 更新全局变量
        global EPOCH, BATCH
        EPOCH = args.epochs
        BATCH = args.batch_size
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')
        print(f'训练参数:')
        print(f'  - 训练轮数: {EPOCH}')
        print(f'  - 批次大小: {BATCH}')
        print(f'  - BERT模型路径: {args.bert_path}')
        print(f'  - 数据集范围: {args.start_dataset} - {args.end_dataset}')
        print(f'  - 学习率: {args.learning_rate}')
        print(f'  - 模型保存目录: {args.save_dir}')
        print("="*50)
        
        # 初始化tokenizer和模型
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        
        # 循环训练数据集
        for i in range(args.start_dataset, args.end_dataset + 1):
            try:
                print(f"\n开始训练数据集 {i}")
                print("="*50)
                
                # 为每个数据集创建新的模型
                model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=2)
                model.to(device)
                
                # 加载当前数据集
                train_dataset = PHQ9Dataset(f'dataset/processed_datasets/train/dataset_{i}.json', tokenizer)
                val_dataset = PHQ9Dataset(f'dataset/processed_datasets/val/dataset_{i}.json', tokenizer)
                
                print(f'数据集 {i}:')
                print(f'  - 训练集大小: {len(train_dataset)}')
                print(f'  - 验证集大小: {len(val_dataset)}')
                
                # 创建数据加载器
                train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH)
                
                # 训练模型
                train_model(model, train_loader, val_loader, device, i)
                
                print(f"数据集 {i} 训练完成")
                print("="*50)
                
                # 清理内存
                del model, train_dataset, val_dataset, train_loader, val_loader
                clear_gpu_memory()
                
            except Exception as e:
                print(f"处理数据集 {i} 时出错: {str(e)}")
                print(traceback.format_exc())
                continue
            
    except Exception as e:
        print(f"主程序出错: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()




