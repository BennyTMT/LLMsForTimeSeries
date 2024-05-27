export CUDA_VISIBLE_DEVICES="0,1,2"
seq_len=512
percent=100
filename=traffic_simple.txt 

gpu_loc=1
tag_file=main.py
methods_h='multi_linr_trsf multi_patch_attn'
model=DLinear_plus
pre_lens_h="96 192 336 720"
for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
if echo "$method" | grep -q "linr"; then
  lr=0.0005 
else
  lr=0.00005
fi
if echo "$method" | grep -q "single"; then
  bs=8192
else
  bs=32
fi
python $tag_file \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id 'traffic_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --method $method \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $bs \
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
    --itr 1 \
    --model $model \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done
