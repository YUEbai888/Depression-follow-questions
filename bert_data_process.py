"""
处理dataset中的PHQ-9数据集  一个list 包含多个字典 每个字典包含post_title, post_text, annotations
需要构造9个数据集 
每个数据集的x是post_title + post_text 在构造数据集x时 需要判断总token是否超过512，如果超过，则尽可能平均分成两条数据
x1为post_title + post_text1
x2为post_title + post_text2
每个数据集的label 是annotations中的一个 yes or no 就是 1 or 0
示例

"""
import pickle
import json
from typing import List, Dict, Tuple
import numpy as np
from transformers import BertTokenizer

import nltk
from nltk.tokenize import sent_tokenize
import re

import nltk

# 添加本地nltk数据路径（应指向nltk_data根目录） linux
# nltk.data.path.append('/home/your_user/nltk_data') 
# 添加本地nltk数据路径（应指向nltk_data根目录）
nltk.data.path.append(r'D:/code/JIEDAN/Follow-up_Question/pkg/nltk_data')

# 检查并下载punkt和punkt_tab（如果没找到的话）
for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

# 测试是否能正常使用sent_tokenize
from nltk.tokenize import sent_tokenize





MAX_TOKEN = 512
tokenizer = BertTokenizer.from_pretrained("bert")  # 可改为对应模型

def clean_text(text):
    """
    清理英文文本
    - 移除多余的空白字符
    - 确保句子之间有正确的空格
    - 移除特殊字符
    """
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 确保句子之间有正确的空格
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    # 移除特殊字符，但保留基本的标点符号
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

def count_tokens(text):
    return len(tokenizer.encode(text))

def split_text_into_chunks(text, max_tokens=512, overlap=True):
    """
    使用NLTK进行英文句子分割，并将文本分成不超过max_tokens的块
    Args:
        text: 输入文本
        max_tokens: 每个块的最大token数
        overlap: 是否在块之间保留重叠
    Returns:
        文本块列表  a,b,c,d,e    
    """
    # 清理文本
    text = clean_text(text)
    
    # 使用NLTK进行英文句子分割
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_token_count = 0

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:  # 跳过空句子
            continue
            
        sentence_token_count = count_tokens(sentence)

        # 如果加上当前句子超过限制，就切断
        if current_token_count + sentence_token_count > max_tokens:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # 保留上下文：重复最后一句或第一句
                if overlap:
                    current_chunk = [current_chunk[-1], sentence]
                    current_token_count = count_tokens(current_chunk[0]) + sentence_token_count
                else:
                    current_chunk = [sentence]
                    current_token_count = sentence_token_count
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_token_count

    # 添加最后一个 chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_phq9_data(data: List[Dict]) -> List[Tuple[List[str], List[int]]]:
    """
    处理PHQ-9数据集
    返回9个数据集的列表，每个数据集包含(x, y)对
    """
    # 初始化9个数据集
    datasets = [[] for _ in range(9)]
    
    for item in data:
        title = item['post_title']
        text = item['post_text']
        annotations = item['annotations']
        
        # 合并标题和文本
        full_text = f"{title} {text}"
        
        # 使用新的分割函数分割文本
        text_chunks = split_text_into_chunks(text, max_tokens=MAX_TOKEN - count_tokens(title), overlap=True)
        
        # 处理每个问题
        for i, (_, label) in enumerate(annotations):
            # 将yes/no转换为1/0
            y = 1 if label.lower() == 'yes' else 0
            
            # 为每个文本块添加标签
            for chunk in text_chunks:
                # 合并标题和文本
                full_chunk = f"{title} {chunk}"
                datasets[i].append((full_chunk, y))
    
    return datasets

# 分别保存这九个数据集
def save_datasets(datasets: List[Tuple[List[str], List[int]]], output_dir: str):
    """
    保存处理后的数据集，每个数据集都划分成训练集和验证集
    Args:
        datasets: 9个数据集的列表
        output_dir: 输出目录
    """
    import os
    import random
    import numpy as np
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    # 设置随机种子，确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 保存每个数据集
    for i, dataset in enumerate(datasets):
        x, y = zip(*dataset)
        x = list(x)
        y = list(y)
        
        # 计算划分点
        total_size = len(x)
        train_size = int(0.8 * total_size)
        
        # 创建索引列表并打乱
        indices = list(range(total_size))
        random.shuffle(indices)
        
        # 划分训练集和验证集
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 准备训练集数据
        train_data = {
            'x': [x[idx] for idx in train_indices],
            'y': [y[idx] for idx in train_indices]
        }
        
        # 准备验证集数据
        val_data = {
            'x': [x[idx] for idx in val_indices],
            'y': [y[idx] for idx in val_indices]
        }
        
        # 保存训练集
        train_file = os.path.join(output_dir, 'train', f'dataset_{i+1}.json')
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        # 保存验证集
        val_file = os.path.join(output_dir, 'val', f'dataset_{i+1}.json')
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        print(f'数据集 {i+1}:')
        print(f'  - 训练集大小: {len(train_indices)}')
        print(f'  - 验证集大小: {len(val_indices)}')
        print(f'  - 已保存到: {train_file} 和 {val_file}')

def main():
    # 读取原始数据
    with open('./dataset/primate_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理数据
    datasets = process_phq9_data(data)
    
    # 保存数据集
    save_datasets(datasets, 'dataset/processed_datasets')

if __name__ == "__main__":
    main()
