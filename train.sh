#!/usr/bin/env bash

# test the best hyperparameters found in the cifar 10 tests

python train.py --dataset mnist \
                --architecture lenet5 \
                --gpu-id 1 \
                --executions 1 \
                --batch-size 128 \
                --training-method sgd,adam,adagrad,rmsprop \
                --learning-method constant \
                --epochs 2 \
                --sgd-lr 0.05 \
                --sgd-momentum 0.9 \
                --sgd-nesterov False \
                --rmsprop-lr 0.0005 \
                --adam-lr 0.0005 \
                --adam-amsgrad True \
                --adagrad-lr 0.05 \
                --results-dir best_results_vgg19
