export CUDA_VISIBLE_DEVICES="0,1,2"
seq_len=512

 
percent=100

gpu_loc=0
#    bash ./scripts/simple/traffic.sh   04 

model=DLinear_plus
filename=traffic_simple.txt 
pre_lens_h="96 192 336 720"
methods_h='single_linr multi_decp_trsf single_linr_decp multi_linr_trsf multi_patch_attn multi_patch_decp'
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
  bs=16
fi
python main.py \
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
