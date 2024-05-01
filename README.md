# TS_Models

## Is LLM truly necessary in time series analysis?

### In this version, I did not clean up the code to preserve its original state during my runtim; this is to avoid removing any potential bugs that might be present in experiments (As shown in our meeting)

## Setup Environment

The Environment in these two eperiments is not complicated, just follow one of original repo, it will be fine.

## Run LLaTA experiments
     cd ./LLaTA

### For ETT experiments :

    sh scripts/long_term_forecasting/ETTh_GPT2.sh

    sh scripts/long_term_forecasting/ETTm_GPT2.sh
    
### For other datasets, such as traffi:
   
    sh scripts/long_term_forecasting/traffic.sh 

### (Please change "itt" parameters in script, if you only want to test)


## Run OFA experiments
    cd ./OFA

### For ETT experiments :
   
  bash ./scripts/ETTh_GPT2.sh   
  
  bash ./scripts/ETTm_GPT2.sh
  
### For other datasets, such as illness:

    bash ./scripts/illness.sh 

### For simple metheds, such as illness : 

   bash ./scripts/simple/illness.sh  (for other dataset)

   bash ./scripts/ETTh_simple.sh (for ETTh)

   bash ./scripts/ETTm_simple.sh (for ETTm)
   
### (Please change "itt" parameters in script, if you only want to test)

