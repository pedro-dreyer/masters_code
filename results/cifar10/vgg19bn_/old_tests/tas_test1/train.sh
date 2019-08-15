#!/usr/bin/env bash

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --gpu-id 0 \
                --executions 2 \
                --training-method sgd \
                --sgd-lr 0.05 \
                --sgd-momentum 0.9 \
                --sgd-nesterov True \
                --learning-method tas \
                --tas-alpha  10,25,50 \
                --tas-beta  0.4,0.5,0.6 \
                --tas-gamma  0.02 \
                --epochs 100 \
                --reduce-train-set 1 \
                --combine-datasets False
