#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'text' \
        --method 'text' \
        --train \
        --data_mode 'binary-class' \
        --save_results \
        --seed $seed \
        --gpu_id '0' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-base-uncased' \
        --config_file_name 'text_bert_bi' \
        --results_file_name 'text_bert_bi.csv'
    done
done