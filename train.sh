# 训练脚本
# python train.py \
#     --train_file=train_and_parallel_and_chatgpt_and_unlabel_text.txt \
#     --val_file_en=dev_en.txt \
#     --val_file_ar=dev_ar.txt \
#     --val_file_ru=dev_ru.txt \
#     --val_file_tr=dev_tr.txt \
#     --lr=1e-5 \
#     --epochs=10 \
#     --batch_size=16 \
#     --model_path=./model/twitter-xlm-roberta-base \
#     --output_dir=./train_and_parallel_and_chatgpt_and_unlabel_best_model \
#     --do_train \
#     --adam_beta1=0.9 \
#     --adam_beta2=0.98 \
#     --adam_epsilon=1e-8 \
#     --warmup_proportion=0. \
#     --weight_decay=0.01 \
#     --save_criterion=all

# 测试脚本
python train.py \
    --val_file_en=dev_en.txt \
    --val_file_ar=dev_ar.txt \
    --val_file_ru=dev_ru.txt \
    --val_file_tr=dev_tr.txt \
    --model_path=./model/twitter-xlm-roberta-base \
    --output_dir=./checkpoint \
    --do_test