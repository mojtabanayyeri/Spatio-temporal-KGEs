#!/bin/bash 
source set_env.sh
python run.py \
            --dataset DBPedia5 \
            --model sFourDELoc \
            --rank 100 \
            --regularizer N3 \
            --reg 0.001 \
            --optimizer Adagrad \
            --max_epochs 500 \
            --patience 15 \
            --valid 10 \
            --batch_size 100 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.1 \
            --gamma 0.0 \
            --bias learn \
            --dtype single \
            --double_neg \
            --cuda_n 2 \
            --dataset_type quintuple_Loc
