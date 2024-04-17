export CUDA_VISIBLE_DEVICES="0,1,2"
train_epochs=100

# dplab04 

filename=randomInit3.txt
model=GPT4TS
percent=100
gpu_loc=0

# ETTh1 
# inp_len=336
# for pred_len in 96 192 336 720;
# do
# for taskType in  'randomInit';
# do
# python main.py \
#     --root_path ./datasets/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id 'ETTh1_'$inp_len'_'$pred_len'_'$taskType \
#     --data ett_h \
#     --seq_len $inp_len \
#     --label_len 168 \
#     --pred_len $pred_len \
#     --batch_size 512 \
#     --lradj type4 \
#     --learning_rate 0.00005 \
#     --train_epochs $train_epochs \
#     --decay_fac 0.5 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --freq 0 \
#     --patch_size 16 \
#     --stride 8 \
#     --percent 100 \
#     --gpt_layer 6 \
#     --itr 3 \
#     --model $model \
#     --tmax 20 \
#     --cos 1 \
#     --pretrain 0 \
#     --gpu_loc $gpu_loc \
#     --is_gpt 1 \
#     --save_file_name $filename
# done
# done

# ETTh2 
# seq_len=336
# for pred_len in 96 192 336 720;
# do
# for taskType in 'randomInit' ;
# do
# python main.py \
#     --root_path ./datasets/ETT-small/ \
#     --data_path ETTh2.csv \
#     --model_id 'ETTh2_'$seq_len'_'$pred_len'_'$taskType \
#     --data ett_h \
#     --seq_len $seq_len \
#     --label_len 168 \
#     --pred_len $pred_len \
#     --batch_size 512 \
#     --decay_fac 0.5 \
#     --learning_rate 0.00005 \
#     --train_epochs $train_epochs \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 1 \
#     --enc_in 7 \
#     --c_out 7 \
#     --freq 0 \
#     --patch_size 16 \
#     --stride 8 \
#     --percent 100 \
#     --gpt_layer 6 \
#     --itr 3 \
#     --model $model \
#     --cos 1 \
#     --tmax 20 \
#     --pretrain 0 \
#     --is_gpt 1 \
#     --gpu_loc $gpu_loc \
#     --save_file_name $filename
# done
# done


# # # ETTm1 
# filename=randomInit2.txt
# model=GPT4TS
# gpu_loc=1
# percent=100
# seq_len=512
# for pred_len in  96 192 336 720;
# do  
# for taskType in  'randomInit' ;
# do
# python main.py \
#     --root_path ./datasets/ETT-small/ \
#     --data_path ETTm1.csv \
#     --model_id 'ETTm1_'$seq_len'_'$pred_len'_'$taskType \
#     --data ett_m \
#     --seq_len $seq_len \
#     --label_len 48 \
#     --pred_len $pred_len \
#     --batch_size 256 \
#     --learning_rate 0.0001 \
#     --train_epochs 10 \
#     --decay_fac 0.75 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --freq 0 \
#     --patch_size 16 \
#     --stride 16 \
#     --percent $percent \
#     --gpt_layer 6 \
#     --itr 3 \
#     --pretrain 0 \
#     --gpu_loc $gpu_loc \
#     --model $model \
#     --cos 1 \
#     --is_gpt 1 \
#     --save_file_name $filename
# done
# done

filename=randomInit3.txt
# # ETTm2  720 336
gpu_loc=2
seq_len=512
for pred_len in 720;
do
for taskType in  'randomInit';
do
python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id 'ETTm2_'$seq_len'_'$pred_len'_'$taskType \
    --data ett_m \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 16 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --pretrain 0 \
    --model $model \
    --cos 1 \
    --gpu_loc $gpu_loc \
    --is_gpt 1 \
    --save_file_name $filename
done
done