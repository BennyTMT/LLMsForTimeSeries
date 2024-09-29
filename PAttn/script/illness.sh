export CUDA_VISIBLE_DEVICES="0,1,2"

seq_len=104
percent=100
gpu_loc=0 
model=PAttn
methods_h='PAttn'

tag_file=main.py
 

filename=Illness_simple.txt 
pre_lens_h="24 36 48 60"
for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
lr=0.00005
bs=8
python $tag_file\
    --root_path ./datasets/illness/ \
    --data_path national_illness.csv \
    --model_id 'Illness_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --method $method \
    --pred_len $pred_len \
    --batch_size $bs \
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
    --itr 1 \
    --model $model \
    --is_gpt 1 \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done
