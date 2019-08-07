#!/usr/bin/env bash

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --gpu-id 1 \
                --executions 2 \
                --training-method sgd \
                --sgd-lr 0.1,0.05 \
                --sgd-momentum 0.9 \
                --sgd-nesterov True \
                --learning-method tas \
		--tas-alpha 25\
		--tas-beta 0.5\
		--tas-gamma 0.02,0.01 \
                --epochs 100 \
                --reduce-train-set 1 \
                --combine-datasets False \
		--batch-size 512
