#!/usr/bin/env bash

# test the best hyperparameters found in the cifar 10 tests

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --gpu-id 1 \
                --executions 3 \
                --batch-size 128 \
                --training-method sgd \
                --learning-method constant \
                --epochs 100 \
                --sgd-lr 2,1,0.5,0.25,0.05,0.01,0.001 \
                --sgd-momentum 0.9\
                --sgd-nesterov True \
		--base-seed 42 \
                --results-dir sgd
