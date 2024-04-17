export CUDA_VISIBLE_DEVICES="0,1,2"

model=GPT4TS
gpu_loc=0

# 96 192 336 
# Electricity
# seq_len=96
# for pred_len in 720; 
# do
# for eval_target in 'ori' 'dropAttn_keepWE' 'Attn_to_Linear'  'Attn_to_Attn'; 
# do 
# python run.py \
#     --root_path ./datasets/electricity/ \
#     --data_path electricity.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id Electricity$model'_'$seq_len'_'$pred_len'_'$eval_target \
#     --data custom \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size 32 \
#     --learning_rate 0.0005 \
#     --train_epochs 20 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --gpt_layer 6 \
#     --itr 1 \
#     --model $model \
#     --cos 1 \
#     --tmax 10 \
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 5 \
#     --task_loss smooth_l1 \
#     --distill_loss smooth_l1 \
#     --logits_loss smooth_l1 \
#     --gpu $gpu_loc \
#     --log_fine_name other_data_attn.txt
# echo '====================================================================================================================='
# done
# done

# Traffic
# seq_len=96
# for pred_len in 96 192 336 720; 
# do
# for eval_target in 'ori' 'dropAttn_keepWE' 'Attn_to_Linear'  'Attn_to_Attn'; 
# do 
# python run.py \
#     --root_path ./datasets/traffic/ \
#     --data_path traffic.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id Traffic_$model'_'$seq_len'_'$pred_len'_'$eval_target \
#     --data custom \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size 8 \
#     --learning_rate 0.0005 \
#     --train_epochs 10 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --gpt_layer 6 \
#     --itr 1 \
#     --model $model \
#     --cos 1 \
#     --tmax 10 \
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 5 \
#     --task_loss smooth_l1 \
#     --distill_loss smooth_l1 \
#     --logits_loss smooth_l1 \
#     --gpu $gpu_loc \
#     --log_fine_name other_data_attn2.txt
# echo '====================================================================================================================='
# done
# done


# 0.20502 0.25019
#  336 720
gpu_loc=0
# # Weather 'ori' 
seq_len=96
for lr in 0.0001; 
do
for pred_len in 96 192 336 720; 
do
for eval_target in 'dropAttn_keepWE' 'Attn_to_Linear'  'Attn_to_Attn'; 
do 
python run.py \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id Weather_$model'_'$seq_len'_'$pred_len'_'$eval_target \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 192 \
    --learning_rate $lr \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --lradj type3 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --task_loss smooth_l1 \
    --distill_loss smooth_l1 \
    --logits_loss smooth_l1 \
    --gpu $gpu_loc \
    --log_fine_name other_data_attn_weather_lr2.txt
echo '====================================================================================================================='
done
done
done 