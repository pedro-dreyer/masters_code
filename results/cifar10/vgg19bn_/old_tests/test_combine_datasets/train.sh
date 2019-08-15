#!/usr/bin/env bash

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --gpu-id 0 \
                --executions 20 \
                --training-method sgd \
                --sgd-lr 0.01 \
                --learning-method constant \
                --epochs 10 \
                --reduce-train-set 1\
                --combine-datasets True,False

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

