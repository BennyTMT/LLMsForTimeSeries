export CUDA_VISIBLE_DEVICES="0,1,2,3"

percent=100
patience=10 
tag_file=main.py

inp_len_h=96
pre_lens_h="96 192 336 720"
filename_h=ETTh_simple.txt 

model=PAttn
methods_h='PAttn'

gpu_loc=0
itt=1
lr=0.00005

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
python $tag_file \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id 'ETTh1_'$inp_len_h'_'$pred_len'_simple_'$method \
    --data ett_h \
    --seq_len $inp_len_h \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 512 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 100 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --patience $patience \
    --percent 100 \
    --gpt_layer 6 \
    --itr $itt \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --method $method \
    --gpu_loc $gpu_loc \
    --is_gpt 1 \
    --save_file_name $filename_h
done
done

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
python $tag_file \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id 'ETTh2_'$inp_len_h'_'$pred_len'_simple_'$method \
    --data ett_h \
    --seq_len $inp_len_h \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 512 \
    --decay_fac 0.5 \
    --learning_rate $lr \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 1 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --patience $patience \
    --itr $itt \
    --model $model \
    --cos 1 \
    --method $method \
    --gpu_loc $gpu_loc \
    --tmax 20 \
    --pretrain 1 \
    --is_gpt 1 \
    --save_file_name $filename_h
done
done
