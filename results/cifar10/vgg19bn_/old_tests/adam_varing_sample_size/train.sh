#!/usr/bin/env bash

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --initial-learning-rate 0.01 \
                --gpu-id 1 \
                --executions 1 \
                --training-method adam \
                --learning-method constant \
                --adam-beta1 0.9 \
                --adam-beta2 0.999  \
                --epochs 100 \
                --reduce-train-set 0.8,0.6,0.4,0.2

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

