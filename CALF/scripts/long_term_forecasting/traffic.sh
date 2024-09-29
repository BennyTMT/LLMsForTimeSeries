export CUDA_VISIBLE_DEVICES="0,1,2"
model=GPT4TS
gpu_loc=2

methods_h="ori dropAttn_keepWE llm_to_attn llm_to_trsf"
filename='traffic_.txt'
seq_len=96
itt=3
# 
#   sh scripts/long_term_forecasting/traffic.sh 
for pred_len in 96 192 336 720; 
do
for eval_target in $methods_h; 
do 
if [ "$eval_target" = 'ori' ]; then
    lr=0.0005
    bs=8
else
    lr=0.001
    bs=32
fi
echo $pred_len"_"$bs"_"$lr
python run.py \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id Traffic_$seq_len'_'$pred_len'_'$eval_target \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $bs \
    --learning_rate $lr \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --gpt_layer 6 \
    --itr $itt \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --task_loss smooth_l1 \
    --distill_loss smooth_l1 \
    --logits_loss smooth_l1 \
    --gpu $gpu_loc \
    --log_fine_name $filename
echo '====================================================================================================================='
done
done



# methods_h="dropAttn_keepWE llm_to_attn llm_to_trsf"
# filename='traffic_tc.txt'
# seq_len=96
# itt=1
# lrs='0.00005 0.0001 0.001'
# bss='8 64'
# #   sh scripts/long_term_forecasting/traffic.sh 
# for pred_len in 96 336; 
# do
# for eval_target in $methods_h; 
# do 
# for lr in $lrs;
# do
# for bs in $bss;
# do
# echo $pred_len"_"$bs"_"$lr
# python run.py \
#     --root_path ./datasets/traffic/ \
#     --data_path traffic.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id Traffic_$seq_len'_'$pred_len'_'$eval_target"_"$bs"_"$lr \
#     --data custom \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size $bs \
#     --learning_rate $lr \
#     --train_epochs 10 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --gpt_layer 6 \
#     --itr $itt \
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
#     --log_fine_name $filename
# echo '====================================================================================================================='
# done
# done
# done
# done