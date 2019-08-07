#!/usr/bin/env bash

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --gpu-id 0 \
                --executions 1 \
                --training-method sgd \
                --sgd-lr 0.01,0.05,0.1,0.3 \
                --sgd-momentum 0,0.3,0.9 \
                --sgd-nesterov False \
                --learning-method constant \
                --epochs 40 \
                --reduce-train-set 1 \
                --combine-datasets False

#export dataset=cifar100
#for model in vgg19bn_
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps #| tee artifacts/$run.$dataset.$model.$exps.txt
#done

#export dataset=imagenet2012
#for model in resnet34
#do
#  export exps="baseline"
#  python $run.py -d $dataset -lm $model -exps $exps -x 1 #| tee artifacts/$run.$dataset.$model.$exps.txt
#done

