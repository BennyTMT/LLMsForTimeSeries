# Are Language Models Actually Useful for Time Series Forecasting?
[Papar Link](https://arxiv.org/pdf/2406.16964) (Neurips 2024 Spotlight 🌟)

In this work we showed that despite the recent popularity of LLMs in time series forecasting (TSF) they do not appear to meaningfully improve performance. A simple baseline, "PAttn," was proposed, which outperformed most LLM-based TSF models.

Authors:[Mingtian Tan](https://www.linkedin.com/in/mingtian-tan-822a182a3/),[Mike A. Merrill](https://mikemerrill.io/),[Vinayak Gupta](https://gvinayak.github.io/),[Tim Althoff](https://homes.cs.washington.edu/~althoff/),[Thomas Hartvigsen](https://www.tomhartvigsen.com/)

## Overview
A great deal of recent work in time series analysis has focused on adapting **pretrained large language models (LLMs)** to forecast, classify, and detect anomalies in time series. These papers posit that language models, being advanced models for sequential dependencies in text, may generalize to the sequential dependencies in time series data. This hypothesis is unsurprising given the popularity of language models in machine learning research writ large.
However, direct connections between language modeling and time series forecasting remain largely undefined.
To what extent is language modeling **really** beneficial for traditional time series tasks? 

After a series of ablation studies on three recent and popular LLM-based time series forecasting methods, we find that **removing the LLM component or replacing it with a basic attention layer** does not degrade the forecasting results---in most cases the results even improved. Additionally, we proposed **PAttn**, demonstrating that patching and attention structures perform comparably to state-of-the-art LLM-based forecasters.

![Ablations/PAttn](pic/ablations.png)


Our goal is not to imply that language models will never be useful for time series. In fact, recent works point to many exciting and promising ways that language and time series interact, like [time series reasoning](https://github.com/behavioral-data/TSandLanguage) and [social understanding](https://github.com/chengjunyan1/SocioDojo).

## Dataset 📖
You can access the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view), then place the downloaded contents under ./dataset

## Setup 🔧
Three different popular LLM-based TSF methods were included in our ablation approach. You might want to follow the corresponding repos, [OneFitsAll](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All), [Time-LLM](https://github.com/KimMeen/Time-LLM), and [CALF](https://github.com/Hank0626/CALF), to set up the environment respectivly. For the ''PAttn'' method, the environment from any of the above repos is compatible.


## PAttn: 
     bash ./scripts/simple/ETTh_simple.sh (for ETTh)
     bash ./scripts/simple/ETTm_simple.sh (for ETTm)
     
### simple methods, other dataset, for example Weather

     bash ./scripts/simple/weather.sh 

####  Note that, the method and in script, "multi_patch_attn", means "PAttn". And the mehtod name, "multi_linr_trsf", means "LTrsf"

## Ablations
     
#### Run Ablations on CALF (ETT) :
     
    cd ./CALF
    sh scripts/long_term_forecasting/ETTh_GPT2.sh
    sh scripts/long_term_forecasting/ETTm_GPT2.sh
    
    sh scripts/long_term_forecasting/traffic.sh 
    (For other datasets, such as traffic)

#### Run Ablations on OneFitsAll (ETT) :
     cd ./OFA
     bash ./scripts/ETTh_GPT2.sh   
     bash ./scripts/ETTm_GPT2.sh

     bash ./scripts/illness.sh 
     (For other datasets, such as illness)

#### Run Ablations on  Time-LLM (ETT) 
     cd ./Time-LLM-exp
     bash ./scripts/train_script/TimeLLM_ETTh1.sh
     bash ./scripts/train_script/TimeLLM_ETTm1.sh 

     bash ./scripts/train_script/TimeLLM_Weather.sh
     (For other datasets, such as Weather)

#### (To run ablations on other datasets, please change the name as shown in example.)

     

