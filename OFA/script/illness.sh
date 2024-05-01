export CUDA_VISIBLE_DEVICES=0

seq_len=104
percent=100

model=GPT4TS
filename=illness.txt 
pre_lens_h="96 192 336 720"
methods_h="ori removeLLM llm_to_attn llm_to_trsf"
gpu_loc=0

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
if [ $method == 'ori' ]; then
    lr=0.0001
else
    lr=0.00005
fi
python main.py \
    --root_path ./datasets/illness/ \
    --data_path national_illness.csv \
    --model_id 'Illness_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 24 \
    --stride 2 \
    --all 1 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --is_gpt 1 \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done

