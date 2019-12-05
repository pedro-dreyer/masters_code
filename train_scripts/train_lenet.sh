#!/usr/bin/env bash

# test the best hyperparameters found in the cifar 10 tests

python train.py --dataset mnist \
                --architecture lenet5 \
                --gpu-id 1 \
                --executions 1 \
                --batch-size 1024 \
                --training-method sgd \
                --learning-method constant \
                --epochs 2 \
                --sgd-lr 2,1,0.5,0.25,0.05,0.01,0.001 \
                --sgd-momentum 0\
                --sgd-nesterov False \
		--adam-lr 0.005,0.001,0.0005,0.0003,0.0001,0.00005 \
		--rmsprop-lr 0.01,0.005,0.001,0.0005,0.0003,0.0001 \
		--rmsprop-centered True \
		--adagrad-lr 0.1,0.05,0.01,0.0075,0.005 \
		--adam-amsgrad True,False \
                --results-dir new_tests
