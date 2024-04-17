# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="0,1,2"
filename=other_data_results.txt 
percent=100
model=GPT4TS

#  bash ./scripts/other_data_attn.sh
# # Electricity 
# train 5715405
# val   814377
# test  1657965
#  192 336 720
filename=other_data_results_Electricity.txt 
seq_len=336
gpu_loc=0
# 'ori' 'dropAttn_keepWE' 'Attn_to_Linear' 'Attn_to_Attn'
for pred_len in 96;
do
for taskType in 'Attn_to_Linear';
do
python main.py \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id 'Electricity_'$seq_len'_'$pred_len'_'$taskType \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 8192 \
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
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done

#  bash ./scripts/other_data_attn.sh
# filename=other_data_results_illnes_linear.txt 
# gpu_loc=1
# # # illnes 
# model=GPT4TS
# percent=100
# seq_len=104
# # 'dropAttn_keepWE' 'Attn_to_Linear' 'Attn_to_Attn'
# for lr in 0.001 0.0005;
# do
# for pred_len in 24 36 48 60;
# do
# for taskType in 'Attn_to_Linear';
# do
# python main.py \
#     --root_path ./datasets/illness/ \
#     --data_path national_illness.csv \
#     --model_id 'Illness_'$seq_len'_'$pred_len'_'$taskType \
#     --data custom \
#     --seq_len $seq_len \
#     --label_len 18 \
#     --pred_len $pred_len \
#     --batch_size 16 \
#     --learning_rate $lr \
#     --train_epochs 10 \
#     --decay_fac 0.75 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --freq 0 \
#     --patch_size 24 \
#     --stride 2 \
#     --all 1 \
#     --percent $percent \
#     --gpt_layer 6 \
#     --itr 3 \
#     --model $model \
#     --is_gpt 1 \
#     --gpu_loc $gpu_loc  \
#     --save_file_name $filename
# done
# done
# done 

# 10420718
# 1431782
# 2942006
# 24 hours

   
#        bash ./scripts/other_data_attn.sh
# gpu_loc=7
# filename=other_data_results_traffic.txt 
# # traffic
# # 96 192 336  
# # 'ori' 'dropAttn_keepWE' 'Attn_to_Linear' 'Attn_to_Attn'
# seq_len=96
# model=GPT4TS
# for pred_len in 96
# do
# for taskType in  'Attn_to_Attn';
# do
# python main.py \
#     --root_path ./datasets/traffic/ \
#     --data_path traffic.csv \
#     --model_id 'traffic_'$seq_len'_'$pred_len'_'$taskType \
#     --data custom \
#     --seq_len $seq_len \
#     --label_len 48 \
#     --pred_len $pred_len \
#     --batch_size 8192 \
#     --learning_rate 0.001 \
#     --train_epochs 10 \
#     --decay_fac 0.75 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --freq 0 \
#     --patch_size 16 \
#     --stride 8 \
#     --all 1 \
#     --percent $percent \
#     --gpt_layer 6 \
#     --itr 3 \
#     --model $model \
#     --patience 3 \
#     --cos 1 \
#     --tmax 10 \
#     --is_gpt 1 \
#     --gpu_loc $gpu_loc  \
#     --save_file_name $filename
# done
# done

#  bash ./scripts/other_data_attn.sh
# gpu_loc=0
# # 'ori' 'dropAttn_keepWE'  'Attn_to_Attn'
# filename=other_data_results_weather_atl.txt 
# # weather
# seq_len=336
# model=GPT4TS
# echo 'weather dataset'
# for pred_len in 192 336 720
# do
# for taskType in 'Attn_to_Linear';
# do
# python main.py \
#     --root_path ./datasets/weather/ \
#     --data_path weather.csv \
#     --model_id 'weather_'$seq_len'_'$pred_len'_'$taskType \
#     --data custom \
#     --seq_len $seq_len \
#     --label_len 48 \
#     --pred_len $pred_len \
#     --batch_size 512 \
#     --learning_rate 0.0001 \
#     --train_epochs 10 \
#     --decay_fac 0.9 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --freq 0 \
#     --lradj type3 \
#     --patch_size 16 \
#     --stride 8 \
#     --percent $percent \
#     --gpt_layer 6 \
#     --itr 3 \
#     --model $model \
#     --is_gpt 1 \
#     --gpu_loc $gpu_loc  \
#     --save_file_name $filename
# done
# done
