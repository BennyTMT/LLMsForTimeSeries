export CUDA_VISIBLE_DEVICES="0,1,2"
seq_len=512

model=GPT4TS
filename=traffic.txt 
percent=100
pre_lens_h="96 192 336 720"
methods_h="ori removeLLM llm_to_attn llm_to_trsf"
# methods_h="llm_to_trsf ori"
gpu_loc=2

#    bash ./scripts/traffic.sh 

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
if [ $method == 'ori' ]; then
    lr=0.001
else
    lr=0.0005
fi
python main.py \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id 'traffic_'$seq_len'_'$pred_len'_'$method'_ofa' \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 8192 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --all 1 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done
