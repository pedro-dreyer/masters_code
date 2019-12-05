#!/usr/bin/env bash

# test the best hyperparameters found in the cifar 10 tests
                
python train.py --dataset mnist \
                --architecture lenet5 \
                --gpu-id 1 \
                --executions 3 \
                --batch-size 1024 \
                --training-method sgd \
                --learning-method tas \
                --epochs 100 \
                --sgd-lr 0.05 \
                --sgd-momentum 0.9 \
                --sgd-nesterov False \
                --tas-alpha 25 \
                --tas-beta 0.7 \
                --tas-gamma 0.02 \
		--base-seed 42 \
                --results-dir best_results_vgg19
