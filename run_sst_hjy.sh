#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run_sst.py \
    --gpu 2 --seed 753 \
    --model bert-base-uncased \
    --dataset coco \
    --train_filename 'datasets/CoT/snli/snli_original_add_ml_distribution_4500_19rules.txt' \
    --val_filename 'datasets/sci_chatgpt/test_data/snli_1.0_test_output.txt' \
    --num_labels 3 \
    --epochs 25 --batch_size 32 --learning_rate 3e-5 \
    --save_name 'hjy_snli_original_add_ml_distribution_4500_19rules_s753_test' \
