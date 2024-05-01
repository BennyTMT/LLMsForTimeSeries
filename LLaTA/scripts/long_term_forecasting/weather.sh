export CUDA_VISIBLE_DEVICES="0,1,2"
model=GPT4TS
gpu_loc=1
# ori

methods_h="ori dropAttn_keepWE llm_to_attn llm_to_trsf"
filename='weather.txt'
seq_len=96
itt=3
#   sh scripts/long_term_forecasting/weather.sh 
for pred_len in  96 192 336 720; 
do
for eval_target in $methods_h; 
do 
if [ "$eval_target" = 'ori' ]; then
    lr=0.0005
    bs=64
else
    lr=0.001
    bs=256
fi  
python run.py \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id Weather'_'$seq_len'_'$pred_len'_'$eval_target \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $bs \
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
    --itr $itt \
    --model $model \
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


methods_h="dropAttn_keepWE llm_to_attn llm_to_trsf"
filename='weather_tc.txt'
seq_len=96
itt=1
lrs='0.01'
bss='256'
#   sh scripts/long_term_forecasting/weather.sh 
for pred_len in  96 192 336 720; 
do
for eval_target in $methods_h; 
do 
for lr in $lrs;
do
for bs in $bss;
do
python run.py \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id Weather'_'$seq_len'_'$pred_len'_'$eval_target"_"$bs"_"$lr \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $bs \
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
    --itr $itt \
    --model $model \
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
done
done

