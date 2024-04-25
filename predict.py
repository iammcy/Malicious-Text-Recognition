import sys
import torch
import os
import argparse
import logging
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch
import os
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Predict")


def test_loop(args, dataloader, model):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
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


# 读取数据
def read_data(file):
    dataset_dict = {'text':[]}
    data = open(file).read().split('\n') # 拆分样本
    data = list(filter(None, data)) # 过滤空值
    dataset_dict['text'] = data
    return dataset_dict


# 定义数据集
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

if __name__ == '__main__':
    # 超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file", default=None, type=str, required=True)
    parser.add_argument("--predict_result", default=None, type=str, required=True)
    parser.add_argument("--model_path", default="./model/twitter-xlm-roberta-base", type=str)
    parser.add_argument("--best_model", default="./checkpoint/best_model.bin", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_labels", default=2, type=int)
    args = parser.parse_args()

    # 读取预测数据
    dataset = read_data(args.predict_file)

    # 加载基础模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    # 编码
    encodings = tokenizer(dataset['text'], truncation=True, padding=True, max_length=512)
    # 创建dataset
    predict_data = TextDataset(encodings)

    logger.info(f"Num predict examples - {len(predict_data)}")

    dataloader = DataLoader(predict_data, batch_size=args.batch_size, shuffle=False)

    # 定义模型
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels)

    # 预测
    predictions = predict(args, dataloader, model, tokenizer)

    # 输出预测文件
    with open(args.predict_result, "w") as f:
        for pred, text in zip(predictions, dataset['text']):
            f.write(f"{pred}|{text}\n")