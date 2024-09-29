export CUDA_VISIBLE_DEVICES="0,1,2"
file_name='ETTh_drop.txt'
seq_len=96
bootstrap_eval=0
model=GPT4TS
methods_h="ori dropAttn_keepWE llm_to_attn llm_to_trsf"
pre_lens_h="96 192 336 720"
gpt_loc=0
itt=5
#     sh scripts/long_term_forecasting/ETTh_GPT2.sh
echo $bootstrap_eval
for pred_len in $pre_lens_h; 
do
for eval_target in $methods_h; 
do 
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1'_test_'$seq_len'_'$pred_len'_'$eval_target \
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
    --bootstrap_eval $bootstrap_eval \
    --itr $itt \
    --gpu $gpt_loc \
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
for pred_len in $pre_lens_h;
do
for eval_target in $methods_h; 
do 
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh2'_'$seq_len'_'$pred_len'_'$eval_target \
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
    --itr $itt \
    --model $model \
    --gpu $gpt_loc \
    --bootstrap_eval $bootstrap_eval \
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