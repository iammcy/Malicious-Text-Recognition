from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import recall_score, precision_score
from tqdm.auto import tqdm
from sacrebleu.metrics import BLEU
import torch
import os
import argparse
import logging
import numpy as np
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("LM")


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


def read_parallel_text(data_path, file):
    data = open(os.path.join(data_path, file)).read().split('\n') # 拆分样本
    data = list(map(convert_parallel, data))
    data = list(filter(None, data)) # 过滤None值
    return data


class TRANS(Dataset):
    def __init__(self, parallel_text):
        self.data = self.load_data(parallel_text)
    
    def load_data(self, parallel_text):
        Data = []
        for source, target in parallel_text:
            Data.append({'source':source, 'target':target})
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class PredTRANS(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['source'])
        batch_targets.append(sample['target'])
    batch_data = tokenizer(
        batch_inputs, 
        padding=True, 
        max_length=512,
        truncation=True, 
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets, 
            padding=True, 
            max_length=512,
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        # batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data


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

def test_loop(args, dataloader, model, tokenizer):
    preds, labels = [], []
    bleu = BLEU()

    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=512,
            ).cpu().numpy()
            label_tokens = batch_data["labels"].cpu().numpy()

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

            preds += [pred.strip() for pred in decoded_preds]
            labels += [[label.strip()] for label in decoded_labels]
    return bleu.corpus_score(preds, labels).score

def train(args, train_dataloader, dev_dataloader, model, tokenizer):
    """ Train the model """
    t_total = len(train_dataloader) * args.epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
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

    best_bleu = 0.
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}\n" + 30 * "-")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch)
        dev_bleu = test_loop(args, dev_dataloader, model, tokenizer)
        logger.info(f'Dev: BLEU - {dev_bleu:0.4f}')
        if dev_bleu > best_bleu:
            best_bleu = dev_bleu
            # 删除之前的权重文件
            save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
            for save_weight in save_weights:
                os.remove(os.path.join(args.output_dir, save_weight))
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_bleu_{dev_bleu:0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))

    os.rename(os.path.join(args.output_dir, save_weight), os.path.join(args.output_dir, 'best_model.bin'))
    logger.info("Done!")

def test(args, test_dataloader, model, tokenizer, best_weight):
    logger.info('***** Running testing *****')
    logger.info(f'loading weights from {best_weight}...')
    model.load_state_dict(torch.load(os.path.join(args.output_dir, best_weight)))
    
    bleu = test_loop(args, test_dataloader, model, tokenizer)
    logger.info(f'Test: BLEU - {bleu:0.4f}')

def predict(args, dataloader, model, tokenizer):
    results = []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=512,
            ).cpu().numpy()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            results += [pred.strip() for pred in decoded_preds]
    return results


if __name__ == '__main__':
    # 超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--parallel_text_file", default="parallel_text_by_ChatGPT.txt", type=str)
    parser.add_argument("--predict_file", default="predict.txt", type=str)
    parser.add_argument("--source", default="tr", type=str)
    parser.add_argument("--target", default="en", type=str)
    
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    
    parser.add_argument("--model_path", default="./model/opus-mt-ar-en", type=str)
    parser.add_argument("--output_dir", default="./result", type=str)
    
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run eval on the predict set.")
    
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

    args = parser.parse_args()

    # 模型输出文件
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty.')
    # 创建输出文件夹
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    # 设置设备
    args.device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    
    # 1 加载数据
    parallel_text = {}
    all_text = read_parallel_text(args.data_path, args.parallel_text_file)
    parallel_text['en'] = list(map(lambda x: x[0], all_text))
    parallel_text['ar'] = list(map(lambda x: x[1], all_text))
    parallel_text['ru'] = list(map(lambda x: x[2], all_text))
    parallel_text['tr'] = list(map(lambda x: x[3], all_text))

    # 2 创建Dataset与Dataloader并进行编码
    data = TRANS(zip(parallel_text[args.source], parallel_text[args.target]))
    data_num = len(data)
    # 划分训练、验证、测试集
    train_num = int(0.6*data_num)
    val_num = int(0.2*data_num)
    test_num = data_num - train_num - val_num
    train_data, valid_data, test_data = random_split(data, [train_num, val_num, test_num])
    print(f'train set size: {len(train_data)}')
    print(f'valid set size: {len(valid_data)}')
    print(f'test set size: {len(test_data)}')
    print(next(iter(train_data)))

    # 加载基础模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collote_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collote_fn)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collote_fn)

    # 3 定义模型
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(args.device)

    # 4 训练
    best_weight = 'best_model.bin'
    if args.do_train:
        bleu = test_loop(args, test_dataloader, model, tokenizer)
        logger.info(f'Test: BLEU - {bleu:0.4f}')
        train(args, train_dataloader, valid_dataloader, model, tokenizer)
        test(args, test_dataloader, model, tokenizer, best_weight)
    
    # 5 测试评估
    if args.do_test:
        test(args, test_dataloader, model, tokenizer, best_weight)
    
    # 6 预测
    if args.do_predict:
        test_data = open(os.path.join(args.data_path, args.predict_file)).read().split('\n') # 拆分样本
        test_data = list(filter(None, test_data)) # 过滤None值
        test_encodings = tokenizer(
            test_data, 
            padding=True, 
            max_length=512,
            truncation=True, 
        )
        test_dataset = PredTRANS(test_encodings)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        logger.info(f'loading weights from {best_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, best_weight)))

        pred_trans = predict(args, test_dataloader, model, tokenizer)
        results = []
        for source, target in zip(test_data, pred_trans):
            results.append({
                args.source: source,
                args.target: target
            })
        with open(os.path.join(args.data_path, f'unlabel_{args.source}_to_{args.target}.txt'), 'w', encoding='utf-8') as f:
            for exapmle_result in results:
                f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')