import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import re
import nltk
from nltk.tokenize import sent_tokenize
import os
import logging
import json
from typing import List, Dict, Any
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# 加载配置文件
def load_config() -> Dict[str, Any]:
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("配置文件不存在，使用默认配置")
        return {
            "nltk_data_path": "D:/code/JIEDAN/Follow-up_Question/pkg/nltk_data",
            "bert_model_path": "bert",
            "t5_model_path": "t5_final",
            "max_token": 512,
            "max_questions": 15
        }

# 加载配置
CONFIG = load_config()

# 设置NLTK数据路径
nltk.data.path.append(CONFIG["nltk_data_path"])

# 检查并下载必要的NLTK数据
for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

# 常量定义
MAX_TOKEN = CONFIG["max_token"]
MAX_QUESTIONS = CONFIG["max_questions"]

# 初始化tokenizer
try:
    bert_tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model_path"])
    t5_tokenizer = AutoTokenizer.from_pretrained(CONFIG["t5_model_path"])
except Exception as e:
    logging.error(f"初始化tokenizer失败: {str(e)}")
    raise

def clean_text(text):
    """清理文本"""
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 确保句子之间有正确的空格
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    # 移除特殊字符，但保留基本的标点符号
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

def count_tokens(text):
    """计算文本的token数量"""
    return len(bert_tokenizer.encode(text))

class ModelManager:
    """模型管理器，负责按需加载模型"""
    def __init__(self):
        self.bert_models = {}  # 存储已加载的BERT模型
        self.t5_model = None   # T5模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_bert_model(self, model_id):
        """按需加载BERT模型"""
        try:
            if model_id not in self.bert_models:
                model_path = f"bert_models/dataset_{model_id}/best_model_threshold_0.5"
                if os.path.exists(model_path):
                    # logging.info(f"正在加载模型 {model_id}，路径: {model_path}")
                    model = BertForSequenceClassification.from_pretrained(model_path)
                    model.to(self.device)
                    model.eval()
                    self.bert_models[model_id] = model
                    # logging.info(f"模型 {model_id} 加载成功")
                else:
                    error_msg = f"模型目录 {model_path} 不存在"
                    logging.error(error_msg)
                    raise FileNotFoundError(error_msg)
            return self.bert_models[model_id]
        except Exception as e:
            logging.error(f"加载模型 {model_id} 时出错: {str(e)}")
            raise
    
    def load_t5_model(self):
        """加载T5模型"""
        if self.t5_model is None:
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5_final")
            self.t5_model.to(self.device)
            self.t5_model.eval()
        return self.t5_model
    
    def clear_bert_models(self):
        """清理已加载的BERT模型"""
        for model in self.bert_models.values():
            del model
        self.bert_models.clear()
        torch.cuda.empty_cache()

def process_with_bert(text_chunks: List[str], model_manager: ModelManager, device: str) -> List[List[int]]:
    """使用BERT模型处理文本块"""
    results = []
    
    for chunk in text_chunks:
        try:
            chunk_results = []
            inputs = bert_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=MAX_TOKEN)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 依次加载并使用每个BERT模型
            for model_id in range(1, 10):
                try:
                    model = model_manager.load_bert_model(model_id)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)
                        prediction = torch.argmax(probs, dim=1).item() # 0或1  
                        chunk_results.append(prediction)
                except Exception as e:
                    logging.error(f"处理模型 {model_id} 时出错: {str(e)}")
                    chunk_results.append(0)
            # 1,1,0,0,0,0,0,0,1
            results.append(chunk_results)
            
            # 处理完一个chunk后清理模型
            model_manager.clear_bert_models()
            
        except Exception as e:
            logging.error(f"处理文本块时出错: {str(e)}")
            results.append([0] * 9)  # 出错时返回全0结果
    
    return results

def generate_questions(text_chunks, model_manager, device='cuda' if torch.cuda.is_available() else 'cpu', max_seq=MAX_QUESTIONS):
    """使用T5模型生成问题"""
    all_questions = []
    
    # 加载T5模型
    t5_model = model_manager.load_t5_model()
    
    for chunk in text_chunks:
        # 准备输入文本
        input_text = f"generate questions: {chunk}"
        inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_TOKEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成问题
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs['input_ids'],
                max_length=MAX_TOKEN,
                num_return_sequences=max_seq,  # 使用传入的max_seq
                do_sample=True,
                top_p=0.9,
                temperature=1.0,  # 添加温度参数
                no_repeat_ngram_size=2,  # 避免重复
                early_stopping=True
            )
        
        # 解码生成的问题
        questions = [t5_tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
        
        # 过滤问题
        filtered_questions = []
        for q in questions:
            if "fuck" not in q.lower():
                filtered_questions.append(q)
        
        all_questions.extend(filtered_questions)
    
    # 确保返回的问题数量不超过max_seq
    return all_questions[:max_seq]

def truncate_text(text, max_tokens=512):
    """直接截断文本到指定token长度"""
    tokens = bert_tokenizer.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return bert_tokenizer.decode(truncated_tokens)
    return text

def process_text(text, max_tokens=512):
    """处理输入文本：清理并截断"""
    # 清理文本
    cleaned_text = clean_text(text)
    
    # 如果文本较短，直接返回
    if count_tokens(cleaned_text) <= max_tokens:
        return [cleaned_text]
    
    # 如果文本较长，直接截断
    return [truncate_text(cleaned_text, max_tokens)]

def validate_input(text: str) -> bool:
    """验证输入文本"""
    if not text or not isinstance(text, str):
        return False
    if len(text.strip()) < 10:  # 文本太短
        return False
    return True

def main():
    try:
        # 初始化模型管理器
        model_manager = ModelManager()
        
        bert_labels_list = [
            'How often have you been bothered by feeling bad about yourself - that you are a failure or have let yourself or your family down?', 
            'Bothered by feeling down, depressed, or hopeless?', 
            'How often have you been bothered by feeling tired or having little energy?',
            'How often have you been bothered by little interest or pleasure in doing things?', 
            'How often have you been bothered by moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?', 
            'How often have you been bothered by poor appetite or overeating?', 
            'How often have you been bothered by thoughts that you would be better off dead or of hurting yourself in some way', 
            'How often have you been bothered by trouble concentrating while reading newspaper or watching television?', 
            'Have been bothered by trouble falling or staying asleep, or sleeping too much?'
        ]
        
        while True:
            try:
                # 获取用户输入
                text = input("\n请输入文本（输入'q'退出）：")
                if text.lower() == 'q':
                    break
                
                # 验证输入
                if not validate_input(text):
                    print("输入文本无效，请重新输入（至少10个字符）")
                    continue
                
                start_time = time.time()
                
                # 处理文本（清理并截断）
                text_chunks = process_text(text)
                
                # BERT处理
                bert_results = process_with_bert(text_chunks, model_manager, model_manager.device)
                
                # 检查是否所有BERT模型都返回1
                all_ones = all(all(result == 1 for result in chunk_results) for chunk_results in bert_results)
                if all_ones:
                    print("\n这段话包括九个问题")
                    continue
                
                zero_indices = [i for i, val in enumerate(bert_results[0]) if val == 0]
                zero_label_values = [bert_labels_list[i] for i in zero_indices]
                
                print("\n未回答的问题：")
                print("-" * 50)
                for i, question in enumerate(zero_label_values, 1):
                    print(f"{i}. {question}")
                print("-" * 50)
                
                # 确保max_seq不超过MAX_QUESTIONS
                max_seq = min(len(zero_label_values), MAX_QUESTIONS)
                if max_seq == 0:
                    print("\n所有问题都已回答，无需生成新问题")
                    continue

                # T5生成问题
                questions = generate_questions(text_chunks, model_manager, model_manager.device, max_seq)
                
                # 输出生成的问题
                print("\n生成的问题：")
                print("-" * 50)
                for i, q in enumerate(questions, 1):
                    print(f"{i}. {q}")
                print("-" * 50)
                
                end_time = time.time()
                logging.info(f"处理完成，耗时: {end_time - start_time:.2f}秒")
                
            except Exception as e:
                logging.error(f"处理过程中出错: {str(e)}")
                print(f"处理出错: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"程序运行出错: {str(e)}")
        print(f"程序运行出错: {str(e)}")

if __name__ == "__main__":
    main()
