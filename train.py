from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score, precision_score
from tqdm.auto import tqdm
import torch
import os
import argparse
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("LM")

# 读取数据
def convert(x):
    if x !='':
        x = x.split('|')
        return [int(x[0]), x[1]]


def read_data(data_path, file):
    dataset_dict = {'text':[], 'labels':[]}
    data = open(os.path.join(data_path, file)).read().split('\n') # 拆分样本
    data = list(map(convert, data)) # 拆分text、label
    data = list(filter(None, data)) # 过滤None值
    data = list(map(list, zip(*data))) # 转置
    dataset_dict['labels'] += data[0]
    dataset_dict['text'] += data[1]
    return dataset_dict


# 定义数据集
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')

    total_loss = 0.
    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/step:>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(args, val_dataloader, model):
    res = {}
    model.eval()
    with torch.no_grad():
        for key, dataloader in val_dataloader.items():
            res[key] = {'true_labels': [], 'predictions': []}
            for batch_data in tqdm(dataloader):
                res[key]['true_labels'] += batch_data['labels']
                batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
                outputs = model(**batch_data)
                logits = outputs.logits
                pred = logits.argmax(dim=-1)
                res[key]['predictions'] += pred.cpu().numpy().tolist()

    # 计算得分
    beta = 0.7
    metrics = {}
    for key, r in res.items():
        metrics[f'{key}_precision'] = precision_score(r['true_labels'], r['predictions'])
        metrics[f'{key}_recall'] = recall_score(r['true_labels'], r['predictions'])
        metrics[f'{key}_score'] = (1+beta**2)*(metrics[f'{key}_precision']*metrics[f'{key}_recall'])/(beta**2*metrics[f'{key}_precision']+metrics[f'{key}_recall'])
    metrics['avg_score'] = (metrics['en_score']+metrics['ar_score']+metrics['ru_score']+metrics['tr_score'])/4
    return metrics


def train(args, train_dataloader, val_dataloader, model, tokenizer):
    """ Train the model """
    t_total = len(train_dataloader) * args.epochs
    
    # Prepare optimizer and schedule (linear warmup and decay)
    args.warmup_steps = int(t_total * args.warmup_proportion)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.lr, 
        betas=(args.adam_beta1, args.adam_beta2), 
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num Epochs - {args.epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))

    best_avg_score = 0.
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}\n" + 30 * "-")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch)
        metrics = test_loop(args, val_dataloader, model)
        
        if args.save_criterion == 'all':
            avg_score = metrics['avg_score']
        else:
            avg_score = metrics[f'{args.save_criterion}_score']
        logger.info(f'Dev: avg score - {avg_score:0.5f}')
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            # 删除之前的权重文件
            save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
            for save_weight in save_weights:
                os.remove(os.path.join(args.output_dir, save_weight))
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_score_{avg_score:0.5f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    
    os.rename(os.path.join(args.output_dir, save_weight), os.path.join(args.output_dir, 'best_model.bin'))
    logger.info("Done!")


def test(args, val_dataloader, model, tokenizer, best_weight):
    logger.info('***** Running testing *****')
    logger.info(f'loading weights from {best_weight}...')
    model.load_state_dict(torch.load(os.path.join(args.output_dir, best_weight)))
    
    metrics = test_loop(args, val_dataloader, model)
    logger.info(metrics)


if __name__ == '__main__':
    # 超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--train_file", default="train.txt", type=str)
    parser.add_argument("--val_file_en", default="dev_en.txt", type=str)
    parser.add_argument("--val_file_ar", default="dev_ar.txt", type=str)
    parser.add_argument("--val_file_ru", default="dev_ru.txt", type=str)
    parser.add_argument("--val_file_tr", default="dev_tr.txt", type=str)
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model_path", default="./model/xlm-roberta-base", type=str)
    parser.add_argument("--output_dir", default="./result", type=str)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--adam_beta1", default=0.9, type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--adam_beta2", default=0.98, type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, 
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training."
    )
    parser.add_argument("--weight_decay", default=0.01, type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument("--save_criterion", default="all", type=str, 
        help="The optimal weight on which index."
    )

    args = parser.parse_args()

    # 模型输出文件
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty.')
    # 创建输出文件夹
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    # 设置设备
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    
    # 1 加载数据
    all_dataset = {}
    all_dataset['train'] = read_data(args.data_path, args.train_file)
    all_dataset['val_en'] = read_data(args.data_path, args.val_file_en)
    all_dataset['val_ar'] = read_data(args.data_path, args.val_file_ar)
    all_dataset['val_ru'] = read_data(args.data_path, args.val_file_ru)
    all_dataset['val_tr'] = read_data(args.data_path, args.val_file_tr)

    # 2 编码并创建Dataset

    # 加载基础模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    train_encodings = tokenizer(all_dataset['train']['text'], truncation=True, padding=True, max_length=512)
    val_en_encodings = tokenizer(all_dataset['val_en']['text'], truncation=True, padding=True, max_length=512)
    val_ar_encodings = tokenizer(all_dataset['val_ar']['text'], truncation=True, padding=True, max_length=512)
    val_ru_encodings = tokenizer(all_dataset['val_ru']['text'], truncation=True, padding=True, max_length=512)
    val_tr_encodings = tokenizer(all_dataset['val_tr']['text'], truncation=True, padding=True, max_length=512)

    train_data = TextDataset(train_encodings, all_dataset['train']['labels'])
    val_data_en = TextDataset(val_en_encodings, all_dataset['val_en']['labels'])
    val_data_ar = TextDataset(val_ar_encodings, all_dataset['val_ar']['labels'])
    val_data_ru = TextDataset(val_ru_encodings, all_dataset['val_ru']['labels'])
    val_data_tr = TextDataset(val_tr_encodings, all_dataset['val_tr']['labels'])
    logger.info(f"Num train examples - {len(train_data)}")
    logger.info(f"Num val_en examples - {len(val_data_en)}")
    logger.info(f"Num val_ar examples - {len(val_data_ar)}")
    logger.info(f"Num val_ru examples - {len(val_data_ru)}")
    logger.info(f"Num val_tr examples - {len(val_data_tr)}")

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_en_dataloader = DataLoader(val_data_en, batch_size=args.batch_size, shuffle=False)
    val_ar_dataloader = DataLoader(val_data_ar, batch_size=args.batch_size, shuffle=False)
    val_ru_dataloader = DataLoader(val_data_ru, batch_size=args.batch_size, shuffle=False)
    val_tr_dataloader = DataLoader(val_data_tr, batch_size=args.batch_size, shuffle=False)
    val_dataloader = {
        'en': val_en_dataloader,
        'ar': val_ar_dataloader,
        'ru': val_ru_dataloader,
        'tr': val_tr_dataloader
    }

    # 3 定义模型
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels).to(args.device)

    # 4 训练
    if args.do_train:
        train(args, train_dataloader, val_dataloader, model, tokenizer)

    # 5 测试评估
    best_weight = 'best_model.bin'
    if args.do_test:
        test(args, val_dataloader, model, tokenizer, best_weight)
