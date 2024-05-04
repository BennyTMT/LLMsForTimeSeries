export CUDA_VISIBLE_DEVICES="0,1,2"
filename=ETTh_drop.txt 
model=GPT4TS
percent=100
itt=5
gpu_loc=0
inp_len=336
run_file=main.py
# run_file=eval_bs.py

#   bash ./scripts/ETTh_GPT2.sh

pre_lens_h="96 192 336 720"
methods_h="ori removeLLM llm_to_attn llm_to_trsf"

# ETTh1
for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
if [ $method == 'ori' ]; then
    lr=0.0001
else
    lr=0.001
fi
echo $method"_"$lr
python $run_file \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id 'ETTh1_test_'$inp_len'_'$pred_len'_'$method'_ofa' \
    --data ett_h \
    --seq_len $inp_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size  256 \
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
    --itr $itt \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --model $model \
    --gpu_loc $gpu_loc \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --save_file_name $filename
done
done

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
if [ $method == 'ori' ]; then
    lr=0.0001
else
    lr=0.001
fi
echo $method"_"$lr
python $run_file \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id 'ETTh2_'$inp_len'_'$pred_len'_'$method'_ofa' \
    --data ett_h \
    --seq_len $inp_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --decay_fac 0.5 \
    --learning_rate 256 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 1 \
    --itr $itt \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --model $model \
    --cos 1 \
    --gpu_loc $gpu_loc \
    --tmax 20 \
    --pretrain 1 \
    --is_gpt 1 \
     --save_file_name $filename
done
done
