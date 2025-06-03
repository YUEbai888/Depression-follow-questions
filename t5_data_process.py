"""
处理需要输入t5进行微调的数据，构造数据集
primate_dataset.json  添加前缀"generate questions:"以指示生成任务。
label中包含所有没有回答的问题，并不是一个问题
"""
import json
import os
import logging
from tqdm import tqdm
import random
import numpy as np
from transformers import AutoTokenizer,T5Tokenizer
from difflib import SequenceMatcher
MAX_TOKEN = 512

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
labels_list=['How often have you been bothered by feeling bad about yourself - that you are a failure or have let yourself or your family down?', 
             'Bothered by feeling down, depressed, or hopeless?', 
             'How often have you been bothered by feeling tired or having little energy?',
               'How often have you been bothered by little interest or pleasure in doing things?', 
               'How often have you been bothered by moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?', 
               'How often have you been bothered by poor appetite or overeating?', 
               'How often have you been bothered by thoughts that you would be better off dead or of hurting yourself in some way', 
               'How often have you been bothered by trouble concentrating while reading newspaper or watching television?', 
             'Have been bothered by trouble falling or staying asleep, or sleeping too much?']


def count_tokens(text):
    return len(tokenizer.encode(text))

tokenizer = T5Tokenizer.from_pretrained("t5")  # 可改为对应模型

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

def split_text_into_chunks(text, max_tokens=512, overlap=True):
    """
    使用NLTK进行英文句子分割，并将文本分成不超过max_tokens的块
    Args:
        text: 输入文本
        max_tokens: 每个块的最大token数
        overlap: 是否在块之间保留重叠
    Returns:
        文本块列表
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




# 配置日志
def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 't5_data_process.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_data(data_path, output_dir, train_ratio=0.8, seed=42):
    """
    处理数据并分割为训练集和验证集
    
    Args:
        data_path: 原始数据路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        seed: 随机种子
    """
    logger = setup_logging()
    logger.info("开始处理数据...")
    
    try:
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        train_dir = os.path.join(output_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理数据
        processed_data = []
        for item in tqdm(data, desc="处理数据"):
            try:
                # 获取标题和文本
                title = item['post_title']
                text = item['post_text']
                
                # 计算标题的token数
                title_tokens = count_tokens(title)
                # 计算每个chunk可用的最大token数
                max_chunk_tokens = MAX_TOKEN - title_tokens
                max_chunk_tokens = max_chunk_tokens - count_tokens("generate questions: ")
                
                # 切分文本
                text_chunks = split_text_into_chunks(text, max_tokens=max_chunk_tokens, overlap=True)
                
                # 获取标签（所有回答为"no"的问题） 的index 然后在labels_list中找到映射的值 作为label
                labels = []
                for i, annotation in enumerate(item['annotations']):
                    if annotation[1].lower() == 'no':
                        if i < len(labels_list):  # 确保索引在有效范围内
                            labels.append(labels_list[i])
                
                # 如果有标签，则为每个文本块和每个标签创建单独的数据项
                if labels:
                    for chunk in text_chunks:
                        # 构建输入文本
                        input_text = f"generate questions: {title} {chunk}"
                        # 为每个标签创建单独的数据项
                        for label in labels:
                            processed_data.append({
                                'x': input_text,
                                'y': label  # 每个数据项只包含一个标签
                            })
            
            except Exception as e:
                logger.error(f"处理数据项时出错: {str(e)}")
                continue
        
        # 随机打乱数据
        random.shuffle(processed_data)
        
        # 分割训练集和验证集
        train_data = processed_data
        
        # 保存处理后的数据
        train_output = {
            'x': [item['x'] for item in train_data],
            'y': [item['y'] for item in train_data]
        }
        

        
        # 保存训练集
        train_path = os.path.join(train_dir, 't5_dataset.json')
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_output, f, ensure_ascii=False, indent=2)

        
        logger.info(f"数据处理完成:")
        logger.info(f"总数据量: {len(processed_data)}")
        logger.info(f"训练集大小: {len(train_data)}")
        logger.info(f"训练集保存至: {train_path}")
        
    except Exception as e:
        logger.error(f"数据处理过程出错: {str(e)}")
        raise

def main():
    # 数据路径
    primate_dataset_path = "./dataset/primate_dataset.json"
    output_dir = "./dataset/processed_t5_datasets"
    
    # 处理数据
    process_data(primate_dataset_path, output_dir)

if __name__ == "__main__":
    main()

