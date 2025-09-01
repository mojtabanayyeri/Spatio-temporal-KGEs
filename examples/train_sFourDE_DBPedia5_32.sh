#!/bin/bash 
source set_env.sh
python run.py \
            --dataset DBPedia5 \
            --model sFourDE \
            --rank 100 \
            --regularizer N3 \
            --reg 0.00000000001 \
            --optimizer Adagrad \
            --max_epochs 500 \
            --patience 15 \
            --valid 5 \
            --batch_size 100 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.1 \
            --gamma 0.0 \
            --bias learn \
            --dtype single \
            --double_neg \
            --cuda_n 1
