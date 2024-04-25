# 游戏跨语言恶意内容识别
比赛链接：https://gss.tencent.com/competition/2024/index.htm
> 该项目为本人nlp入门小白的决赛提交答案
## 问题建模
本赛题背景为游戏跨语言恶意内容识别，即文本模态风控。对于该赛题，可以主要有两种建模方向，一是直接作为序列分类问题，其可以通过提取多语言序列表征再进行分类；二是建模成序列自回归预测问题，其可以通过大语言模型+prompt+后处理判断。考虑到训练时间、推断效率、游戏领域下标记数据集的限制，本项目将其建模为文本二分类问题。序列分类器搭建采用`transformers`库中的`AutoModelForSequenceClassification`

## 问题分析与挑战
对于本赛题，其提供了高质量的英文文本标记训练数据，因此可以通过该训练集获取在英文文本上表现较好的游戏文本风控模型，但是其在其他语言上表现较差，并且没有其他语言的高质量标记数据。

基于赛题复赛提供的基于ChatGPT生成的标注数据，通过分析其生成的标注数据发现，基于`train.txt`训练好的风控模型在英文验证集上精确率达到0.75，召回率达到0.83，而在ChatGPT标注数据上召回率升高，精确率降低，即说明ChatGPT标注的召回率较低，精确率较高。所以其生成的标注数据标记为1的大概率是1，标记为0的有可能是1或0；

基于赛题复赛提供的基于ChatGPT生成的平行语料，其包含了游戏特定领域下的英文与其他语言的译文，可以基于该语料微调开源的翻译模型，使其适应于游戏场景。因此，基于微调后的翻译模型可以将已有数据集的其他语言翻译成英文，再利用英文文本上的风控模型进行预测打伪标签

## 方案
- 利用`train.txt`训练一个英文上表现好的文本风控分类模型，其在英文验证集上得分为0.775
- 利用英文风控分类模型对平行语料`parallel_text_by_ChatGPT.txt`英文打标，其可以同时获取4种语言的正样本，对于负样本则以正负样本比1:1进行采样，得到平行语料标记数据集`parallel_text_labeled.txt`
- 基于平行语料`parallel_text_by_ChatGPT.txt`训练ar2en、ru2en、tr2en三个翻译模型
- 对于`labeled_text_by_ChatGPT.txt`中标记为1的数据将其保留，对于标记为0的数据进行以下处理：
    - 对于英文，直接利用以上英文风控模型打标召回，并对标记为0数据进行正负样本比1:1采样
    - 对于其他语言，则比英文多了一个翻译的前置步骤，利用以上微调的翻译模型进行翻译，再打标召回
    - 因此得到ChatGPT重标记数据集`relabeled_text_ChatGPT.txt`
- 对于无标注数据集`unlabel_text.txt`，其处理方法与`labeled_text_by_ChatGPT.txt`一致，得到打标后的训练数据集`unlabel_text_labeled.txt`
- 将`train.txt`、`parallel_text_labeled.txt`、`relabeled_text_ChatGPT.txt`、`unlabel_text_labeled.txt`合并得到最终的训练集`train_and_parallel_and_chatgpt_and_unlabel_text.txt`
- 利用最终的训练集训练一个文本二分类模型

## 基础模型选取
- 跨语言的文本分类基础模型采用：`twitter-xlm-roberta-base`：
    - hugging face的[cardiffnlp/twitter-xlm-roberta-base](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base)
- 文本翻译基础模型采用：`Helsinki-NLP`：
    - hugging face的Turkish翻译到English [Helsinki-NLP/opus-mt-tr-en](https://huggingface.co/Helsinki-NLP/opus-mt-tr-en)
    - hugging face的Arabic翻译到English [Helsinki-NLP/opus-mt-ar-en](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en)
    - hugging face的Russian翻译到English [Helsinki-NLP/opus-mt-ru-en](https://huggingface.co/Helsinki-NLP/opus-mt-ru-en)

## 数据构造
本项目采用赛题提供的训练集：
- 8k 条带标注数据（英语），文件名：train.txt

无标注数据集：
- 4*20k 条无标注数据（每个语种各 20k），文件名：unlabel_text.txt
    - 对于其他语种，利用翻译模型将其翻译成对应英文，利用风险模型打标
    - 负样本按1:1采样

ChatGPT标注：
- 4*5k 条 ChatGPT 标注数据（每个语种各 5k），文件名：labeled_text_by_ChatGPT.txt
    - 标记为1的大概率是1，标记为0的有可能是1或0
    - 先通过微调后的翻译模型将序列翻译成英文再预测打标，召回更多的风险文本
    - 剩下的负样本通过1:1采样作为训练集

平行语料数据：
- 50k 平行语料（以英语为原语言，通过 ChatGPT 翻译获取）文件名：
parallel_text_by_ChatGPT.txt
    - 用途一：利用训练好的模型将对应样本语料打标为伪标签，得到对应四种语言的训练数据
    - 用途二：通过语料文本微调开源语言模型得到适用的翻译模型

本项目采用赛题提供的验证集：
- 4*100 条带标注数据（每个语种各 100），文件名：dev_ar.txt、dev_en.txt、dev_ru.txt、dev_tr.txt

## 训练流程
```sh
# 数据集的所有预处理在preprocess.py
# 翻译模型的训练与预测脚本是translate.sh
# 预处理步骤需产生中间文件，交替执行以上脚本

# 合并训练数据集
cd data
sh concat_train_files.sh

# 运行训练脚本
# 微调后的模型权重保存在checkpoint文件夹下
cd ..
sh train.sh

# 在验证集上的指标得分
# 单独train.txt作为训练集为： 0.6341
# train+平行语料打标作为训练集为：0.7022
# train+平行语料打标+chatgpt重打标作为训练集为：0.7177（最终选用模型权重）
# train+平行语料打标+chatgpt重打标+无标记数据作为训练集为：0.7139
```

## 预测使用
```sh
# predict_file为待预测文件，predict_result为预测结果文件
sh predict.sh {predict_file} {predict_result}
```

## 注意
预测主脚本以"predict.sh"命名并放在项目的根目录下，在项目根目录下执行"sh predict.sh {predict\_file} {predict\_result}"来获取预测结果。其中predict\_file为待预测文件，格式为每一行一条文本"{文本}"，predict\_result为预测结果文件，格式为"{标签}|{文本}"。两个文件的逐行对应。