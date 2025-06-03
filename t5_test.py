from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import json
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq
import torch
import nltk
from rouge import Rouge
import numpy as np
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer


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

# 初始化BLEURT模型和tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


bleurt_model_name = "snapshots/90ce39a8ceb3b9a046a7788f0005572301c3fe67"
bleurt_tokenizer = BleurtTokenizer.from_pretrained(bleurt_model_name)
bleurt_model = BleurtForSequenceClassification.from_pretrained(bleurt_model_name)
bleurt_model = bleurt_model.to(device)

model_name = 't5_final' 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

questions143_map_path = "./dataset/questions143_map_number.json"
t5_dataset_path = "./dataset/processed_t5_datasets/train/t5_dataset_map_number.json"

# 加载数据
with open(questions143_map_path, 'r', encoding='utf-8') as f:
    questions143_map = json.load(f)
    
with open(t5_dataset_path, 'r', encoding='utf-8') as f:
    t5_dataset = json.load(f)
# 初始化ROUGE评分器
rouge = Rouge()
def calculate_metrics(generated_texts, references, thresholds=[0.4, 0.5, 0.7]):
    """
    计算 BLEURT 和 ROUGE 分数，并返回在不同阈值下的命中率
    """
    # 计算 BLEURT 分
    bleurt_model.eval()
    with torch.no_grad():
        bleurt_scores = [] # 这里面是生成的n个问题分别对应的最大bleurt分数
        for gen_q in generated_texts:
            inputs = bleurt_tokenizer(references, [gen_q] * len(references), padding=True, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # 计算BLEURT分数
            logits = bleurt_model(**inputs).logits
            # 转为numpy数组，取最大分数
            max_score = logits.squeeze().cpu().numpy().max()
            bleurt_scores.append(max_score)
        
    
    # 计算 ROUGE-L 的f1 分数
    rouge_scores = []
    for gen_q in generated_texts:
        gen_q_score = []
        for ref in references:
            try:
                scores = rouge.get_scores(gen_q, ref)[0]
                gen_q_score.append(scores['rouge-l']['f'])
            except:
                continue
        rouge_scores.append(max(gen_q_score))
        
    # 计算每个阈值下的命中率
    bleurt_counts = [sum(score > t for score in bleurt_scores) for t in thresholds]
    
    rouge_counts = [sum(score > t for score in rouge_scores) for t in thresholds]
    
    return bleurt_counts, rouge_counts



def evaluate_model(maxs_equences=5):
    global model  # 声明使用全局model变量
    model.eval()  # 确保模型在评估模式
    model = model.to(device)  # 确保模型在正确的设备上
    all_bleurt = []
    all_rouge = []
    
    try:
        with torch.no_grad():
            for item in t5_dataset:
                try:
                    # 对输入进行编码
                    inputs = tokenizer(item['x'], return_tensors="pt", max_length=512, truncation=True).to(device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    
                    # 生成预测
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=512,
                        do_sample=True,
                        top_p=0.9,
                        num_return_sequences=5,   # 生成问题的个数
                        temperature=1,
                        no_repeat_ngram_size=2,
                        early_stopping=True,
                        length_penalty=1.0,
                        repetition_penalty=1.2
                    )
                    
                    # 解码生成序列 
                    generated_texts = [tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
                    print("User context----------------------------------------------------:")
                    print(item['x'])
                    print("Generated questions++++++++++++++++++++++++++++++++++++++++++++++++++:")
                    for i, text in enumerate(generated_texts, 1):
                        print(f"{i}. {text}")
                    
                    # 获取未回答问题的子问题作为参考问题
                    reference_questions = []
                    for key in item['y']:
                        key_str = str(key)  # 将key转换为字符串
                        if key_str in questions143_map:
                            reference_questions.extend(questions143_map[key_str])
                    
                    if not reference_questions:
                        print("Warning: No reference questions found for this item")
                        continue
                        
                    # print("Reference questions:")
                    # for i, ref in enumerate(reference_questions, 1):
                    #     print(f"{i}. {ref}")
                    
                    bleurt_counts, rouge_counts = calculate_metrics(generated_texts, reference_questions)
                    all_bleurt.append(bleurt_counts)
                    all_rouge.append(rouge_counts)
                    
                except Exception as e:
                    print(f"Error processing item: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return None, None

    if not all_bleurt or not all_rouge:
        print("Warning: No valid results collected")
        return None, None

    assert len(all_bleurt) == len(all_rouge)
    num_samples = len(all_bleurt)  # 数据集样本数
    num_generated = 5  # 每个样本生成的问题数
    total_questions = num_samples * num_generated  # 总问题数
    
    # 计算平均值
    all_bleurt_total = [sum(row[i] for row in all_bleurt) / total_questions for i in range(len(all_bleurt[0]))]
    all_rouge_total = [sum(row[i] for row in all_rouge) / total_questions for i in range(len(all_rouge[0]))]
    
    return all_bleurt_total, all_rouge_total

if __name__ == "__main__":
    all_bleurt_total, all_rouge_total = evaluate_model()
    if all_bleurt_total is not None and all_rouge_total is not None:
        print("\nBLEURT scores (thresholds: 0.4, 0.5, 0.7):")
        print(all_bleurt_total)
        
        print("\nROUGE scores (thresholds: 0.4, 0.5, 0.7):")
        print(all_rouge_total)
    else:
        print("Evaluation failed to produce valid results")

