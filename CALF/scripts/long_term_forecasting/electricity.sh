export CUDA_VISIBLE_DEVICES="0,1,2"
model=GPT4TS
gpu_loc=2
bootstrap_eval=1
methods_h="ori dropAttn_keepWE llm_to_attn llm_to_trsf"
filename='electricity.txt'
seq_len=96
itt=3
#   sh scripts/long_term_forecasting/electricity.sh 
for pred_len in 96 192 336 720; 
do
for eval_target in $methods_h; 
do 
if [ "$eval_target" = 'ori' ]; then
    lr=0.0005
    bs=32
else
    lr=0.001
    bs=64
fi  
echo $eval_target"_"$pred_len"_"$bs"_"$lr
python run.py \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id Electricity_$seq_len'_'$pred_len'_'$eval_target \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $bs \
    --learning_rate $lr \
    --train_epochs 20 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
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
    --bootstrap_eval $bootstrap_eval \
    --log_fine_name $filename
echo '====================================================================================================================='
done
done


exit
# Adjust parameters : 
filename='electricity_tc.txt'
seq_len=96
itt=1
lrs='0.0001 0.001'
bss='32'
#   sh scripts/long_term_forecasting/electricity.sh 
for pred_len in 96 192 336 720; 
do
for eval_target in $methods_h; 
do 
for lr in $lrs;
do
for bs in $bss;
do
echo $eval_target"_"$pred_len"_"$bs"_"$lr
python run.py \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id Electricity'_'$seq_len'_'$pred_len'_'$eval_target"_"$bs"_"$lr \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $bs \
    --learning_rate $lr \
    --train_epochs 20 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
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
done
done
