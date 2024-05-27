export CUDA_VISIBLE_DEVICES="0,1,2"
model_name=TimeLLM
train_epochs=10
learning_rate=0.1
llama_layers=32

master_port=8871
num_process=3
batch_size=16
d_model=16
d_ff=32

comment='TimeLLM-illness'

itts='0'
train_epochs=25
methods_h="removeLLM llm_to_trsf llm_to_attn"
for method in $methods_h;
do
for itt in $itts;
do
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/illness/  \
  --data_path national_illness.csv \
  --model_id Illness_104_24_$method \
  --model $model_name \
  --data Illness \
  --features M \
  --seq_len 104 \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --itr $itt \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id Illness_104_36_$method \
  --model $model_name \
  --data Illness \
  --features M \
  --seq_len 104 \
  --label_len 0 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
    --itr $itt \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

  accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id Illness_104_48_$method \
  --model $model_name \
  --data Illness \
  --features M \
  --seq_len 104 \
  --label_len 0 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 1 \
    --itr $itt \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

  accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id Illness_104_60_$method \
  --model $model_name \
  --data Illness \
  --features M \
  --seq_len 104 \
  --label_len 0 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --itr $itt \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
done 
done 