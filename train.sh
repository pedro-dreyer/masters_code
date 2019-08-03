#!/usr/bin/env bash

# test TAS with bad parameters

python train.py --dataset cifar100 \
                --architecture vgg19bn_ \
                --gpu-id 0 \
                --executions 2 \
                --training-method sgd \
                --learning-method constant \
                --epochs 2 \
                --sgd-lr 0.25,0.05,0.01,0.001 \
                --sgd-momentum 0.9 \
                --results-dir leaning_rate_testing
