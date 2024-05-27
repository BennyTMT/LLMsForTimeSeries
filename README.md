# LLM for Time Series Ablation Methods

## Is LLM truly useful in time series analysis?

## DataSet
You can access the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view), then place the downloaded contents under ./dataset

## Setup Environment

The Environment in these two eperiments is not complicated, just follow one of original repo, it will be fine.

## Run LLaTA experiments
     cd ./LLaTA

### For ETT experiments :

    sh scripts/long_term_forecasting/ETTh_GPT2.sh

    sh scripts/long_term_forecasting/ETTm_GPT2.sh
    
### For other datasets, such as traffic:
   
    sh scripts/long_term_forecasting/traffic.sh 

### (Please change "itt" parameters in script, if you only want to test)


## Run OFA experiments
    cd ./OFA

### For ETT experiments :
   
     bash ./scripts/ETTh_GPT2.sh   
  
     bash ./scripts/ETTm_GPT2.sh
  
### For other datasets, such as illness:

     bash ./scripts/illness.sh 

## Run Time-LLM exepriments experiments
     cd ./Time-LLM-exp

### For ETT datasets 

     bash ./scripts/train_script/TimeLLM_ETTh1.sh
     bash ./scripts/train_script/TimeLLM_ETTh2.sh 
     bash ./scripts/train_script/TimeLLM_ETTm1.sh 
     bash ./scripts/train_script/TimeLLM_ETTm2.sh 
     
### For other datasets

     bash ./scripts/train_script/TimeLLM_Weather.sh
     bash ./scripts/train_script/TimeLLM_Traffic.sh
     bash ./scripts/train_script/TimeLLM_ECL.sh
     bash ./scripts/train_script/illness.sh

### The original script saved in ./scripts/train_script/train_ori/


### (Please change "itt" parameters in script, if you only want to test)


## For "simple metheds" : 
     
     bash ./scripts/simple/ETTh_simple.sh (for ETTh)

     bash ./scripts/simple/ETTm_simple.sh (for ETTm)
     
### simple methods, other dataset, for example Weather

     bash ./scripts/simple/weather.sh 

## Note that, the method and in script, "multi_patch_attn", means "PAttn". And the mehtod name, "multi_linr_trsf", means "LTrsf"
     

