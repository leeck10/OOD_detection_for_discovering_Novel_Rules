#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run_sst.py \
    --gpu 1 --seed 0 \
    --model bert-base-uncased \
    --dataset coco \
    --train_filename 'datasets/generated_train_set.txt' \
    --val_filename 'datasets/generated_test_set' \
    --num_labels 3 \
    --epochs 25 --batch_size 32 --learning_rate 3e-5 \
    --save_name 'finetuned_15_transformation_rules' \


