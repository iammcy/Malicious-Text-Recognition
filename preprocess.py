import json
import re
import os
import argparse
import torch
import numpy as np
import logging
import random
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Predict")

# 读取数据
def convert(x):
    if x !='':
        x = x.split('|')
        type = x[0]
        text = x[1]
        label = 0
        try:
            start = re.search("{", x[2]).span()[0]
            end = re.search("}", x[2]).span()[1]
            tmp = x[2][start: end]
            result = json.loads(tmp)
        except:
            return None
        for k, v in result.items():
            if v == "true":
                label = 1
                break
        return [type, text, label]


def convert_parallel(x):
    try:
        x = x.split('|')[1]
        x = json.loads(x)
        en = x['original']
        ar = x['Arabic']
        ru = x['Russian']
        tr = x['Turkish']
        if isinstance(en, str) and isinstance(ar, str) and isinstance(ru, str) and isinstance(tr, str):
            return [en, ar, ru, tr]
        else:
            return None
    except:
        return None


def text_preprocess(x):
    try:
        x = x.split('|')
        type = x[0]
        text = x[1]
    except:
        return None
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = t.rstrip('0123456789%')
        new_text.append(t)
    new_text = " ".join(new_text).strip()
    if new_text=='':
        return None
    else:
        return [type, new_text]


def read_labeled_by_chatgpt(data_path, file):
    data = open(os.path.join(data_path, file)).read().split('\n') # 拆分样本
    data = list(map(convert, data)) # 拆分text、label
    data = list(filter(None, data)) # 过滤None值
    return data


def read_unlabeled_text(data_path, unlabeled_file):
    data = open(os.path.join(data_path, unlabeled_file)).read().split('\n') # 拆分样本
    data = list(filter(None, data)) # 过滤None值
    data = list(map(text_preprocess, data))
    data = list(filter(None, data)) # 过滤None值
    return data


def read_parallel_text(data_path, file):
    data = open(os.path.join(data_path, file)).read().split('\n') # 拆分样本
    data = list(map(convert_parallel, data))
    data = list(filter(None, data)) # 过滤None值
    return data


def read_trans_text(data_path, file):
    data = open(os.path.join(data_path, file)).read().split('\n') # 拆分样本
    data = list(filter(None, data)) # 过滤None值
    data = list(map(lambda x: json.loads(x), data))
    return data


# 定义数据集
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def test_loop(args, dataloader, model):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            outputs = model(**batch_data)
            logits = outputs.logits
            predictions += logits.argmax(dim=-1).cpu().numpy().tolist()

    return predictions     


def predict(args, dataloader, model, tokenizer):
    logger.info('***** Running predicting *****')
    logger.info(f'loading weights from {args.best_model}...')
    model.load_state_dict(torch.load(args.best_model))
    predictions = test_loop(args, dataloader, model)
    return predictions


if __name__ == '__main__':
    # 超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--labeled_by_chatgpt_file", default="labeled_text_by_ChatGPT.txt", type=str)
    parser.add_argument("--unlabeled_file", default="unlabel_text.txt", type=str)

    parser.add_argument("--paralled_text_file", default="parallel_text_by_ChatGPT.txt", type=str)
    parser.add_argument("--model_path", default="./model/twitter-xlm-roberta-base", type=str)
    parser.add_argument("--best_model", default="./en_best_model/best_model.bin", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_labels", default=2, type=int)

    args = parser.parse_args()

    # 处理chatgpt标记的文本数据
    # 1. 对于chatgpt标记为1的数据：直接作为正样本
    # 2. 对于chatgpt未识别为1的数据：
    #       1.1. 对于英文文本利用已训练风险模型捞回
    #       1.2. 对于其他语言文本先利用翻译模型转英文，再利用风险模型捞回
    # 3. 剩下负样本采样作为训练集
    '''
    labeled_by_chatgpt = read_labeled_by_chatgpt(args.data_path, args.labeled_by_chatgpt_file)

    labeled_by_chatgpt_en = list(filter(lambda x: x[0]=='en', labeled_by_chatgpt))
    labeled_by_chatgpt_en_pos = list(filter(lambda x: x[2]==1, labeled_by_chatgpt_en))
    labeled_by_chatgpt_en_neg = list(filter(lambda x: x[2]==0, labeled_by_chatgpt_en))
    
    labeled_by_chatgpt_ar = list(filter(lambda x: x[0]=='ar', labeled_by_chatgpt))
    labeled_by_chatgpt_ar_pos = list(filter(lambda x: x[2]==1, labeled_by_chatgpt_ar))
    labeled_by_chatgpt_ar_neg = list(filter(lambda x: x[2]==0, labeled_by_chatgpt_ar))

    labeled_by_chatgpt_ru = list(filter(lambda x: x[0]=='ru', labeled_by_chatgpt))
    labeled_by_chatgpt_ru_pos = list(filter(lambda x: x[2]==1, labeled_by_chatgpt_ru))
    labeled_by_chatgpt_ru_neg = list(filter(lambda x: x[2]==0, labeled_by_chatgpt_ru))

    labeled_by_chatgpt_tr = list(filter(lambda x: x[0]=='tr', labeled_by_chatgpt))
    labeled_by_chatgpt_tr_pos = list(filter(lambda x: x[2]==1, labeled_by_chatgpt_tr))
    labeled_by_chatgpt_tr_neg = list(filter(lambda x: x[2]==0, labeled_by_chatgpt_tr))
    '''

    '''
    # with open(os.path.join(args.data_path, 'labeled_chatgpt_en.txt'), "w") as f:
    #     for _, text, label in labeled_by_chatgpt_en:
    #         f.write(f"{label}|{text}\n")
    with open(os.path.join(args.data_path, 'labeled_chatgpt_en_pos.txt'), "w") as f:
        for _, text, label in labeled_by_chatgpt_en_pos:
            f.write(f"{label}|{text}\n")
    with open(os.path.join(args.data_path, 'labeled_chatgpt_en_neg.txt'), "w") as f:  
        for _, text, label in labeled_by_chatgpt_en_neg:
            f.write(f"{text}\n")
    
    # with open(os.path.join(args.data_path, 'labeled_chatgpt_ar.txt'), "w") as f:
    #     for _, text, label in labeled_by_chatgpt_ar:
    #         f.write(f"{label}|{text}\n")
    with open(os.path.join(args.data_path, 'labeled_chatgpt_ar_pos.txt'), "w") as f:
        for _, text, label in labeled_by_chatgpt_ar_pos:
            f.write(f"{label}|{text}\n")
    with open(os.path.join(args.data_path, 'labeled_chatgpt_ar_neg.txt'), "w") as f:
        for _, text, label in labeled_by_chatgpt_ar_neg:
            f.write(f"{text}\n")
    
    # with open(os.path.join(args.data_path, 'labeled_chatgpt_ru.txt'), "w") as f:
    #     for _, text, label in labeled_by_chatgpt_ru:
    #         f.write(f"{label}|{text}\n")
    with open(os.path.join(args.data_path, 'labeled_chatgpt_ru_pos.txt'), "w") as f:
        for _, text, label in labeled_by_chatgpt_ru_pos:
            f.write(f"{label}|{text}\n")
    with open(os.path.join(args.data_path, 'labeled_chatgpt_ru_neg.txt'), "w") as f:
        for _, text, label in labeled_by_chatgpt_ru_neg:
            f.write(f"{text}\n")
    
    # with open(os.path.join(args.data_path, 'labeled_chatgpt_tr.txt'), "w") as f:
    #     for _, text, label in labeled_by_chatgpt_tr:
    #         f.write(f"{label}|{text}\n")
    with open(os.path.join(args.data_path, 'labeled_chatgpt_tr_pos.txt'), "w") as f:
        for _, text, label in labeled_by_chatgpt_tr_pos:
            f.write(f"{label}|{text}\n")
    with open(os.path.join(args.data_path, 'labeled_chatgpt_tr_neg.txt'), "w") as f:
        for _, text, label in labeled_by_chatgpt_tr_neg:
            f.write(f"{text}\n")
    '''

    '''
    en_neg = list(map(lambda x: x[1], labeled_by_chatgpt_en_neg))
    ar2en = read_trans_text(args.data_path, 'labeled_chatgpt_neg_ar_to_en.txt')
    ar2en_en = list(map(lambda x: x['en'], ar2en))
    ar2en_ar = list(map(lambda x: x['ar'], ar2en))
    ru2en = read_trans_text(args.data_path, 'labeled_chatgpt_neg_ru_to_en.txt')
    ru2en_en = list(map(lambda x: x['en'], ru2en))
    ru2en_ru = list(map(lambda x: x['ru'], ru2en))
    tr2en = read_trans_text(args.data_path, 'labeled_chatgpt_neg_tr_to_en.txt')
    tr2en_en = list(map(lambda x: x['en'], tr2en))
    tr2en_tr = list(map(lambda x: x['tr'], tr2en))
    # 设置设备
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 加载基础模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    # 编码
    en_encodings = tokenizer(en_neg, truncation=True, padding=True, max_length=512)
    ar2en_en_encodings = tokenizer(ar2en_en, truncation=True, padding=True, max_length=512)
    ru2en_en_encodings = tokenizer(ru2en_en, truncation=True, padding=True, max_length=512)
    tr2en_en_encodings = tokenizer(tr2en_en, truncation=True, padding=True, max_length=512)
    # 创建dataset
    en_predict_data = TextDataset(en_encodings)
    ar2en_en_predict_data = TextDataset(ar2en_en_encodings)
    ru2en_en_predict_data = TextDataset(ru2en_en_encodings)
    tr2en_en_predict_data = TextDataset(tr2en_en_encodings)
    en_dataloader = DataLoader(en_predict_data, batch_size=args.batch_size, shuffle=False)
    ar2en_en_dataloader = DataLoader(ar2en_en_predict_data, batch_size=args.batch_size, shuffle=False)
    ru2en_en_dataloader = DataLoader(ru2en_en_predict_data, batch_size=args.batch_size, shuffle=False)
    tr2en_en_dataloader = DataLoader(tr2en_en_predict_data, batch_size=args.batch_size, shuffle=False)
    # 定义模型
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels).to(args.device)
    # 预测打标
    en_predictions = predict(args, en_dataloader, model, tokenizer)
    ar2en_en_predictions = predict(args, ar2en_en_dataloader, model, tokenizer)
    ru2en_en_predictions = predict(args, ru2en_en_dataloader, model, tokenizer)
    tr2en_en_predictions = predict(args, tr2en_en_dataloader, model, tokenizer)
    
    # 负样本采样
    en_predictions = np.array(en_predictions)
    en_pos_idx = np.where(en_predictions==1)[0]
    en_neg_idx = np.where(en_predictions==0)[0]
    en_pos_num = len(en_pos_idx)+len(labeled_by_chatgpt_en_pos)
    en_neg_idx = np.random.choice(a=en_neg_idx, size=min(en_pos_num, len(en_neg_idx)), replace=False)

    ar2en_en_predictions = np.array(ar2en_en_predictions)
    ar2en_en_pos_idx = np.where(ar2en_en_predictions==1)[0]
    ar2en_en_neg_idx = np.where(ar2en_en_predictions==0)[0]
    ar2en_en_pos_num = len(ar2en_en_pos_idx)+len(labeled_by_chatgpt_ar_pos)
    ar2en_en_neg_idx = np.random.choice(a=ar2en_en_neg_idx, size=min(ar2en_en_pos_num, len(ar2en_en_neg_idx)), replace=False)

    ru2en_en_predictions = np.array(ru2en_en_predictions)
    ru2en_en_pos_idx = np.where(ru2en_en_predictions==1)[0]
    ru2en_en_neg_idx = np.where(ru2en_en_predictions==0)[0]
    ru2en_en_pos_num = len(ru2en_en_pos_idx)+len(labeled_by_chatgpt_ru_pos)
    ru2en_en_neg_idx = np.random.choice(a=ru2en_en_neg_idx, size=min(ru2en_en_pos_num, len(ru2en_en_neg_idx)), replace=False)

    tr2en_en_predictions = np.array(tr2en_en_predictions)
    tr2en_en_pos_idx = np.where(tr2en_en_predictions==1)[0]
    tr2en_en_neg_idx = np.where(tr2en_en_predictions==0)[0]
    tr2en_en_pos_num = len(tr2en_en_pos_idx)+len(labeled_by_chatgpt_tr_pos)
    tr2en_en_neg_idx = np.random.choice(a=tr2en_en_neg_idx, size=min(tr2en_en_pos_num, len(tr2en_en_neg_idx)), replace=False)

    # 保存relabeled chatgpt训练数据集
    with open(os.path.join(args.data_path, 'relabeled_text_ChatGPT.txt'), "w") as f:
        for _, text, label in labeled_by_chatgpt_en_pos:
            f.write(f"{label}|{text}\n")
        for text in np.array(en_neg)[en_pos_idx]:
            f.write(f"1|{text}\n")
        for text in np.array(en_neg)[en_neg_idx]:
            f.write(f"0|{text}\n")

        for _, text, label in labeled_by_chatgpt_ar_pos:
            f.write(f"{label}|{text}\n")
        for text in np.array(ar2en_ar)[ar2en_en_pos_idx]:
            f.write(f"1|{text}\n")
        for text in np.array(ar2en_ar)[ar2en_en_neg_idx]:
            f.write(f"0|{text}\n")

        for _, text, label in labeled_by_chatgpt_ru_pos:
            f.write(f"{label}|{text}\n")
        for text in np.array(ru2en_ru)[ru2en_en_pos_idx]:
            f.write(f"1|{text}\n")
        for text in np.array(ru2en_ru)[ru2en_en_neg_idx]:
            f.write(f"0|{text}\n")

        for _, text, label in labeled_by_chatgpt_tr_pos:
            f.write(f"{label}|{text}\n")
        for text in np.array(tr2en_tr)[tr2en_en_pos_idx]:
            f.write(f"1|{text}\n")
        for text in np.array(tr2en_tr)[tr2en_en_neg_idx]:
            f.write(f"0|{text}\n")
    '''


    # 处理无标记文本数据
    # 1.对于英文文本直接利用风险模型打标
    # 2.对于其他语言文本利用翻译模型翻译后再进行打标
    # 3.负样本按正负1:1采样
    unlabeled_text = read_unlabeled_text(args.data_path, args.unlabeled_file)
    unlabeled_en = list(filter(lambda x: x[0]=='en', unlabeled_text))
    unlabeled_en = [x[1] for x in unlabeled_en]
    unlabeled_ar = list(filter(lambda x: x[0]=='ar', unlabeled_text))
    unlabeled_ar = [x[1] for x in unlabeled_ar]
    unlabeled_ru = list(filter(lambda x: x[0]=='ru', unlabeled_text))
    unlabeled_ru = [x[1] for x in unlabeled_ru]
    unlabeled_tr = list(filter(lambda x: x[0]=='tr', unlabeled_text))
    unlabeled_tr = [x[1] for x in unlabeled_tr]
    print(f"en num: {len(unlabeled_en)}; ar num: {len(unlabeled_ar)}; ru num: {len(unlabeled_ru)}; tr num: {len(unlabeled_tr)}")

    # 写入待翻译文本数据中间文件
    # with open(os.path.join(args.data_path, 'unlabel_ar.txt'), "w") as f:
    #     for text in unlabeled_ar:
    #         f.write(f"{text}\n")
    # with open(os.path.join(args.data_path, 'unlabel_ru.txt'), "w") as f:
    #     for text in unlabeled_ru:
    #         f.write(f"{text}\n")
    # with open(os.path.join(args.data_path, 'unlabel_tr.txt'), "w") as f:
    #     for text in unlabeled_tr:
    #         f.write(f"{text}\n")
    ar2en = read_trans_text(args.data_path, 'unlabel_ar_to_en.txt')
    ar2en_en = list(map(lambda x: x['en'], ar2en))
    ar2en_ar = list(map(lambda x: x['ar'], ar2en))
    ru2en = read_trans_text(args.data_path, 'unlabel_ru_to_en.txt')
    ru2en_en = list(map(lambda x: x['en'], ru2en))
    ru2en_ru = list(map(lambda x: x['ru'], ru2en))
    tr2en = read_trans_text(args.data_path, 'unlabel_tr_to_en.txt')
    tr2en_en = list(map(lambda x: x['en'], tr2en))
    tr2en_tr = list(map(lambda x: x['tr'], tr2en))
    # 设置设备
    args.device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    # 加载基础模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    # 编码
    en_encodings = tokenizer(unlabeled_en, truncation=True, padding=True, max_length=512)
    ar2en_en_encodings = tokenizer(ar2en_en, truncation=True, padding=True, max_length=512)
    ru2en_en_encodings = tokenizer(ru2en_en, truncation=True, padding=True, max_length=512)
    tr2en_en_encodings = tokenizer(tr2en_en, truncation=True, padding=True, max_length=512)
    # 创建dataset
    en_predict_data = TextDataset(en_encodings)
    ar2en_en_predict_data = TextDataset(ar2en_en_encodings)
    ru2en_en_predict_data = TextDataset(ru2en_en_encodings)
    tr2en_en_predict_data = TextDataset(tr2en_en_encodings)
    en_dataloader = DataLoader(en_predict_data, batch_size=args.batch_size, shuffle=False)
    ar2en_en_dataloader = DataLoader(ar2en_en_predict_data, batch_size=args.batch_size, shuffle=False)
    ru2en_en_dataloader = DataLoader(ru2en_en_predict_data, batch_size=args.batch_size, shuffle=False)
    tr2en_en_dataloader = DataLoader(tr2en_en_predict_data, batch_size=args.batch_size, shuffle=False)
    # 定义模型
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels).to(args.device)
    # 预测打标
    en_predictions = predict(args, en_dataloader, model, tokenizer)
    ar2en_en_predictions = predict(args, ar2en_en_dataloader, model, tokenizer)
    ru2en_en_predictions = predict(args, ru2en_en_dataloader, model, tokenizer)
    tr2en_en_predictions = predict(args, tr2en_en_dataloader, model, tokenizer)
    
    # 负样本采样
    en_predictions = np.array(en_predictions)
    en_pos_idx = np.where(en_predictions==1)[0]
    en_neg_idx = np.where(en_predictions==0)[0]
    en_pos_num = len(en_pos_idx)
    en_neg_idx = np.random.choice(a=en_neg_idx, size=min(en_pos_num, len(en_neg_idx)), replace=False)

    ar2en_en_predictions = np.array(ar2en_en_predictions)
    ar2en_en_pos_idx = np.where(ar2en_en_predictions==1)[0]
    ar2en_en_neg_idx = np.where(ar2en_en_predictions==0)[0]
    ar2en_en_pos_num = len(ar2en_en_pos_idx)
    ar2en_en_neg_idx = np.random.choice(a=ar2en_en_neg_idx, size=min(ar2en_en_pos_num, len(ar2en_en_neg_idx)), replace=False)

    ru2en_en_predictions = np.array(ru2en_en_predictions)
    ru2en_en_pos_idx = np.where(ru2en_en_predictions==1)[0]
    ru2en_en_neg_idx = np.where(ru2en_en_predictions==0)[0]
    ru2en_en_pos_num = len(ru2en_en_pos_idx)
    ru2en_en_neg_idx = np.random.choice(a=ru2en_en_neg_idx, size=min(ru2en_en_pos_num, len(ru2en_en_neg_idx)), replace=False)

    tr2en_en_predictions = np.array(tr2en_en_predictions)
    tr2en_en_pos_idx = np.where(tr2en_en_predictions==1)[0]
    tr2en_en_neg_idx = np.where(tr2en_en_predictions==0)[0]
    tr2en_en_pos_num = len(tr2en_en_pos_idx)
    tr2en_en_neg_idx = np.random.choice(a=tr2en_en_neg_idx, size=min(tr2en_en_pos_num, len(tr2en_en_neg_idx)), replace=False)

    # 保存unlabel数据打标的训练数据集
    with open(os.path.join(args.data_path, 'unlabel_text_labeled.txt'), "w") as f:
        for text in np.array(unlabeled_en)[en_pos_idx]:
            f.write(f"1|{text}\n")
        for text in np.array(unlabeled_en)[en_neg_idx]:
            f.write(f"0|{text}\n")

        for text in np.array(ar2en_ar)[ar2en_en_pos_idx]:
            f.write(f"1|{text}\n")
        for text in np.array(ar2en_ar)[ar2en_en_neg_idx]:
            f.write(f"0|{text}\n")

        for text in np.array(ru2en_ru)[ru2en_en_pos_idx]:
            f.write(f"1|{text}\n")
        for text in np.array(ru2en_ru)[ru2en_en_neg_idx]:
            f.write(f"0|{text}\n")

        for text in np.array(tr2en_tr)[tr2en_en_pos_idx]:
            f.write(f"1|{text}\n")
        for text in np.array(tr2en_tr)[tr2en_en_neg_idx]:
            f.write(f"0|{text}\n")

    # 处理平行语料数据
    '''
    # 1.抽取英文语料文本
    # 2.利用train.txt上微调的风控模型进行打标
    # 3.对应得到平行语料的伪标签
    # 4.对负样本进行采样
    # 5.保存平行语料训练数据集：parallel_text_labeled.txt
    parallel_text = read_parallel_text(args.data_path, args.paralled_text_file)
    parallel_text_en = list(map(lambda x: x[0], parallel_text))
    parallel_text_ar = list(map(lambda x: x[1], parallel_text))
    parallel_text_ru = list(map(lambda x: x[2], parallel_text))
    parallel_text_tr = list(map(lambda x: x[3], parallel_text))
    
    # 设置设备
    args.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # 加载基础模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    # 编码
    encodings = tokenizer(parallel_text_en, truncation=True, padding=True, max_length=512)
    # 创建dataset
    predict_data = TextDataset(encodings)
    dataloader = DataLoader(predict_data, batch_size=args.batch_size, shuffle=False)
    # 定义模型
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels).to(args.device)
    # 预测打标
    predictions = predict(args, dataloader, model, tokenizer)
    
    # 负样本采样
    predictions = np.array(predictions)
    parallel_text_en = np.array(parallel_text_en)
    parallel_text_ar = np.array(parallel_text_ar)
    parallel_text_ru = np.array(parallel_text_ru)
    parallel_text_tr = np.array(parallel_text_tr)
    pos_idx = np.where(predictions==1)[0]
    neg_idx = np.where(predictions==0)[0]
    pos_num = len(pos_idx)
    neg_num = len(neg_idx)

    neg_idx = np.random.choice(a=neg_idx, size=pos_num, replace=False)

    # 保存平行语料训练数据集   
    with open(os.path.join(args.data_path, 'parallel_text_labeled.txt'), "w") as f:
        for text in parallel_text_en[neg_idx]:
            f.write(f"0|{text}\n")
        for text in parallel_text_en[pos_idx]:
            f.write(f"1|{text}\n")

        for text in parallel_text_ar[neg_idx]:
            f.write(f"0|{text}\n")
        for text in parallel_text_ar[pos_idx]:
            f.write(f"1|{text}\n")

        for text in parallel_text_ru[neg_idx]:
            f.write(f"0|{text}\n")
        for text in parallel_text_ru[pos_idx]:
            f.write(f"1|{text}\n")

        for text in parallel_text_tr[neg_idx]:
            f.write(f"0|{text}\n")
        for text in parallel_text_tr[pos_idx]:
            f.write(f"1|{text}\n")
    '''


