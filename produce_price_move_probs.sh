#!/usr/bin/env bash
PYTHONPATH="/Users/huayongzhou/anaconda3/bin/"
python=${PYTHONPATH}python3.7

export PATH=$PATH:/Users/huayongzhou/dev/win

cd /Users/huayongzhou/dev/win
DATA_FILE=/Users/huayongzhou/dev/win/all_data.csv
MODEL_PKL=/Users/huayongzhou/dev/win/model_output.pkl
PRED_FILE=/Users/huayongzhou/dev/win/predict_data.csv

if [ -f "$DATA_FILE" ]; then
    rm -f $DATA_FILE
fi

if [ -f "$MODEL_PKL" ]; then
    rm -f $MODEL_PKL
fi

if [ -f "$PRED_FILE" ]; then
    rm -f $PRED_FILE
fi

if ls /Users/huayongzhou/dev/win/prediction_from* 1> /dev/null 2>&1; then
    rm -f prediction_from*
fi

$python ~/dev/win/BTC_Greedy_Fear_index.py
$python ~/dev/win/scrap_yield.py
$python ~/dev/win/ML_model.py
$python ~/dev/win/send_email.py
