#!/usr/bin/env bash

# test the best hyperparameters found in the cifar 10 tests

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --gpu-id 0 \
                --executions 3 \
                --batch-size 128 \
                --training-method adagrad \
                --learning-method constant \
                --epochs 100 \
                --adagrad-lr 0.1,0.05,0.01,0.0075,0.005\
		        --base-seed 42 \
                --results-dir adagrad
