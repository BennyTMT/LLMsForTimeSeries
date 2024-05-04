export CUDA_VISIBLE_DEVICES="0,1,2"
filename=ETTm2_drop.txt 
model=GPT4TS
percent=100
itt=5
gpu_loc=1
inp_len=512
run_file=main.py 
#run_file=eval_bs.py

#   bash ./scripts/ETTm_GPT2.sh   
# pre_lens_h="96 192 336 720"
pre_lens_h="96 192 336 720"
methods_h="ori removeLLM llm_to_attn llm_to_trsf"

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
if [ $method == 'ori' ]; then
    lr=0.0001
    bs=512
else
    lr=0.00005
    bs=512
fi
echo $method"_"$lr
python $run_file \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id 'ETTm1_'$inp_len'_'$pred_len'_'$method'_ofa' \
    --data ett_m \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size $bs \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --itr $itt \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 16 \
    --percent $percent \
    --gpt_layer 6 \
    --cos 1 \
    --is_gpt 1 \
    --seq_len $inp_len \
    --learning_rate $lr \
    --model $model \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
if [ $method == 'ori' ]; then
    lr=0.0001
    bs=512
else
    lr=0.00005
    bs=512
fi
echo $method"_"$lr
python $run_file \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id 'ETTm2_'$inp_len'_'$pred_len'_'$method'_ofa' \
    --data ett_m \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size $bs \
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
    --itr $itt \
    --cos 1 \
    --is_gpt 1 \
    --seq_len $inp_len \
    --learning_rate $lr \
    --model $model \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done
