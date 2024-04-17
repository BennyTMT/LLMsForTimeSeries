export CUDA_VISIBLE_DEVICES="0,1,2"
# dropAttn_keepWE , Attn_to_Linear , Attn_to_Attn

model=GPT4TS
# taskType=Attn_to_Attn
percent=100
pred_len=96
lr=0.0001
seq_len=96

for taskType in 'dropAttn_keepWE' 'Attn_to_Linear' 'Attn_to_Attn';
do
python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len $seq_len \
    --pred_len $pred_len \
    --batch_size 256 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 10 \
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
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --save_file_name $taskType.txt
done
# done

# python test_in_validation.py \
#     --root_path ./datasets/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
#     --data ett_h \
#     --seq_len $seq_len \
#     --label_len 168 \
#     --pred_len $pred_len \
#     --batch_size 256 \
#     --lradj type4 \
#     --learning_rate $lr \
#     --train_epochs 10 \
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
#     --percent $percent \
#     --gpt_layer 6 \
#     --itr 3 \
#     --model $model \
#     --tmax 20 \
#     --cos 1 \
#     --is_gpt 1
