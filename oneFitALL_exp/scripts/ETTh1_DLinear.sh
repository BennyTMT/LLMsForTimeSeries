export CUDA_VISIBLE_DEVICES="0,1,2"

# seq_len=336
model=DLinear
gpuLoc=2
# for percent in 100
# do
# for pred_len in 96 192 336 720
# do
# for lr in 0.0001
# do

# 512 0.0005 best
percent=100
pred_len=96
lr=0.0005
seq_len=336 


# seq="32 64 96 144 192 336 448 512 720"
# seq="512"
# for seq_len in $seq;
# do  
# python main.py \
#     --root_path ./datasets/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
#     --data ett_h \
#     --seq_len $seq_len \
#     --label_len $seq_len \
#     --pred_len $pred_len \
#     --batch_size 512 \
#     --lradj type4 \
#     --learning_rate $lr \
#     --train_epochs 10 \
#     --decay_fac 0.5 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --freq 0 \
#     --patch_size 16 \
#     --stride 8 \
#     --percent $percent \
#     --gpt_layer 6 \
#     --itr 3 \
#     --model $model \
#     --tmax 20 \
#     --cos 1 \
#     --is_gpt 1 \
#     --save_file_name DLinear_$seq_len'_'$pred_len'_768_0005.txt' \
#     --gpu_loc $gpuLoc
# done

# python test_in_validation.py \
#     --root_path ./datasets/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
#     --data ett_h \
#     --seq_len $seq_len \
#     --label_len 168 \
#     --pred_len $pred_len \
#     --batch_size 256 \
#     --lradj type4 \
#     --learning_rate $lr \
#     --train_epochs 10 \
#     --decay_fac 0.5 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --freq 0 \
#     --patch_size 16 \
#     --stride 8 \
#     --percent $percent \
#     --gpt_layer 6 \
#     --itr 3 \
#     --model $model \
#     --tmax 20 \
#     --cos 1 \
#     --is_gpt 1


rs="0.1 0.2 0.4 0.6 0.8 1.0"
for train_ratio in $rs;
do
python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len $seq_len \
    --pred_len $pred_len \
    --batch_size 512 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --train_ratio $train_ratio
done