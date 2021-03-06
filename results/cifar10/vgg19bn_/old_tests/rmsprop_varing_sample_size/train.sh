#!/usr/bin/env bash

python train.py --dataset cifar10 \
                --architecture vgg19bn_ \
                --initial-learning-rate 0.0003 \
                --gpu-id 1 \
                --executions 1 \
                --training-method rmsprop \
                --learning-method constant \
                --epochs 100 \
                --reduce-train-set 1,0.8,0.6,0.4,0.2 \
                --combine-datasets False,True

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

