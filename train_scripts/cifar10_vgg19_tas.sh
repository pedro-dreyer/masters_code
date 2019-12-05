#!/usr/bin/env bash

# test the best hyperparameters found in the cifar 10 tests

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --gpu-id 0 \
                --executions 3 \
                --batch-size 128 \
                --training-method sgd \
                --learning-method tas \
		--sgd-lr 0.05 \
		--sgd-momentum 0.9 \
		--tas-alpha 10,25,50 \
		--tas-beta 0.3,0.5,0.7 \
                --epochs 100 \
		--base-seed 42 \
                --results-dir tas
