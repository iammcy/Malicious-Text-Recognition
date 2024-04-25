# 1.已有英文训练集+平行语料打标+chatgpt标记文本的捞回重打标
# cat train.txt parallel_text_labeled.txt relabeled_text_ChatGPT.txt > train_and_parallel_and_chatgpt_text.txt

# 2.在1的基础上+无标记打标数据
cat train.txt parallel_text_labeled.txt relabeled_text_ChatGPT.txt unlabel_text_labeled.txt > train_and_parallel_and_chatgpt_and_unlabel_text.txt
