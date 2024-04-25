# 训练脚本
# python translate.py \
#     --parallel_text_file=parallel_text_by_ChatGPT.txt \
#     --source=tr \
#     --target=en \
#     --lr=1e-4 \
#     --epochs=3 \
#     --batch_size=32 \
#     --model_path=./model/opus-mt-tr-en \
#     --output_dir=./tr2en_best_model \
#     --do_train \
#     --adam_beta1=0.9 \
#     --adam_beta2=0.98 \
#     --adam_epsilon=1e-8 \
#     --warmup_proportion=0. \
#     --weight_decay=0. \

# 测试脚本
python translate.py \
    --predict_file=unlabel_tr.txt \
    --source=tr \
    --target=en \
    --model_path=./model/opus-mt-tr-en \
    --output_dir=./tr2en_best_model \
    --do_predict