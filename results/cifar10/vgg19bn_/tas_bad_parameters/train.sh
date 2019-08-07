#!/usr/bin/env bash

# test TAS with bad parameters

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --gpu-id 0 \
                --executions 2 \
                --training-method sgd\
                --learning-method tas\
                --epochs 30 \
                --sgd-lr 0.05 \
                --sgd-momentum 0.9 \
                --tas-alpha 300 \
                --tas-beta 0.05,0.9 \
                --tas-gamma 0.0025 \
                --results-dir tas_bad_parameters
