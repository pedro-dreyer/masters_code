#!/usr/bin/env bash

# test the best hyperparameters found in the cifar 10 tests

python train.py --dataset cifar10 \
                --architecture resnet18_ \
                --gpu-id 0 \
                --executions 2 \
                --batch-size 1024 \
                --training-method sgd,adam \
                --learning-method constant \
                --epochs 100 \
                --sgd-lr 0.05 \
                --sgd-momentum 0.9 \
                --sgd-nesterov False \
                --rmsprop-lr 0.0005 \
                --adam-lr 0.0005 \
                --adam-amsgrad True \
                --adagrad-lr 0.05 \
                --results-dir best_results_vgg19
