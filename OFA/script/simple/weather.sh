export CUDA_VISIBLE_DEVICES="0,1"

seq_len=512
percent=100
gpu_loc=1

#    bash ./scripts/simple/weather.sh 
# 96 192 336 
model=DLinear_plus
filename=weather_simple1.txt 
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
  bs=1024
else
  bs=128
fi
python main.py \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id 'weather_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --seq_len $seq_len \
    --method $method \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size  $bs \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --lradj type3 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 2 \
    --model $model \
    --is_gpt 1 \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done
