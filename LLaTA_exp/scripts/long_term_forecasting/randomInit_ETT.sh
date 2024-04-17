export CUDA_VISIBLE_DEVICES="0,1,2"

itr=3
file_name='ETT_randomInit1.txt'
model=GPT4TS
seq_len=96
for pred_len in 96 192 336 720; 
do
for eval_target in 'ori' 'randomInit'; 
do 
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len'_'$eval_target \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layer 6 \
    --itr $itr \
    --model $model \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 10 \
    --log_fine_name $file_name
echo '====================================================================================================================='
done
done


seq_len=96
model=GPT4TS
for pred_len in 96 192 336 720;
do
for eval_target in 'ori' 'randomInit'; 
do 
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh2_$model'_'$seq_len'_'$pred_len'_'$eval_target \
    --data ETTh2 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layers 6 \
    --itr $itr \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --log_fine_name $file_name
echo '====================================================================================================================='
done
done 


# file_name='ETT_randomInit2.txt'
# seq_len=96
# model=GPT4TS
# for pred_len in 96 192 336 720;
# do
# for eval_target in 'ori' 'randomInit' ; 
# do
# python run.py \
#     --root_path ./datasets/ETT-small/ \
#     --data_path ETTm1.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id ETTm1_$model'_'$seq_len'_'$pred_len'_'$eval_target \
#     --data ETTm1 \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size 256 \
#     --learning_rate 0.0005 \
#     --lradj type1 \
#     --train_epochs 100 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --gpt_layer 6 \
#     --itr $itr \
#     --model $model \
#     --cos 1 \
#     --tmax 20 \
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 5 \
#     --log_fine_name $file_name
# echo '====================================================================================================================='
# done
# done


# seq_len=96
# model=GPT4TS
# for pred_len in 96 192 336 720;
# do
# for eval_target in 'ori' 'randomInit'; 
# do
# python run.py \
#     --root_path ./datasets/ETT-small/ \
#     --data_path ETTm2.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id ETTm2_$model'_'$seq_len'_'$pred_len'_'$eval_target \
#     --data ETTm2 \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size 256 \
#     --learning_rate 0.0001 \
#     --lradj type1 \
#     --train_epochs 100 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --gpt_layer 6 \
#     --itr $itr \
#     --model $model \
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 5 \
#     --log_fine_name $file_name
# echo '====================================================================================================================='
# done
# done 
