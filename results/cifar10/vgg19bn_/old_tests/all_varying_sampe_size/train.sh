#!/usr/bin/env bash

# the results were obtained using the learning rate for adagrad equal to 0.001 while the correct one was supossed to be 0.01
python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --gpu-id 1 \
                --executions 1 \
                --training-method rmsprop,adam,adagrad,sgd \
                --adam-lr 0.001 \
                --rmsprop-lr 0.01 \
                --adagrad-lr 0.001 \
                --sgd-lr 0.01 \
                --learning-method constant \
                --epochs 100 \
                --reduce-train-set 1,0.8,0.6,0.4,0.2,0.1 \
                --combine-datasets True

