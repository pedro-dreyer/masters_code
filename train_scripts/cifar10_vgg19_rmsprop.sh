#!/usr/bin/env bash

# test the best hyperparameters found in the cifar 10 tests

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --gpu-id 0 \
                --executions 3 \
                --batch-size 128 \
                --training-method rmsprop \
                --learning-method constant \
                --epochs 100 \
                --rmsprop-lr 0.01,0.005,0.001,0.0005,0.0003,0.0001 \
		--rmsprop-centered False,True \
		--base-seed 42 \
                --results-dir rmsprop
