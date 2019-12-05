import argparse
import os
import time
import random
import numpy
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
import torchnet as tnt
import math
import csv
import itertools

from tqdm import tqdm

from argparse import RawTextHelpFormatter

from torch.nn.modules.module import _addindent
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import models
from datasets import ImageFolder

from datetime import datetime


# TODO see what those do 
cudnn.benchmark = False
cudnn.deterministic = True
#cudnn.benchmark = True
#cudnn.deterministic = False

numpy.set_printoptions(formatter={'float': '{:0.4f}'.format})
torch.set_printoptions(precision=4)
pd.set_option('display.width', 160)

dataset_names = ('mnist', 'cifar10', 'cifar100', 'imagenet2012')

local_model_names = sorted(name for name in models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(models.__dict__[name]))

# TODO why is there remote_models?
remote_model_names = sorted(name for name in torchvision_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torchvision_models.__dict__[name]))

parser = argparse.ArgumentParser(description='Train', formatter_class=RawTextHelpFormatter)


# Architecture

# parser.add_argument('-lm', '--local-model', metavar='MODEL', default=None, choices=local_model_names,
#                     help='model to be used: ' + ' | '.join(local_model_names))
# parser.add_argument('-rm', '--remote-model', metavar='MODEL', default=None, choices=remote_model_names,
#                     help='model to be used: ' + ' | '.join(remote_model_names))

parser.add_argument('--architecture', help='architecture to be used')

# Dataset
parser.add_argument('--dataset', metavar='DATA', default=None, choices=dataset_names, nargs='?',
                    help='dataset to be used: ' + ' | '.join(dataset_names))
parser.add_argument('--combine-datasets', default='False', type=str, help='Combine train plus test datasets')
parser.add_argument('--do-validation-set', default='False', type=str, help='Separate train dataset in train plus validation dataset')


# General options
parser.add_argument('--epochs', default='100', type=str, help='number of total epochs to run')
parser.add_argument('--batch-size', default='128', type=str, help='mini-batch size (default: 128)')
parser.add_argument('--test-set-split', default='0.1666667', type=str, help='fraction of dataset to be used as the test set')
parser.add_argument('--validation-set-split', default='0.0', type=str, help='fraction of trainset to be used as the validation set')
parser.add_argument('--reduce-train-set', default='1', type=str, help='Reduce train set by this fraction')

# Training method
parser.add_argument('--training-method', default='adam', type=str,
                    help='select the training method and its parameters. acepted methods adam, adagrad, rmsprop, SGD')

# parser.add_argument('--initial-learning-rate', type=str, help='initial learning rate')

# Trainig method - Hyperparameters

# sgd
parser.add_argument('--sgd-lr', type=str, default='0.01', help='learning rate value for the sgd algorithm')
parser.add_argument('--sgd-momentum', default='0', type=str, help='momentum value for the sgd algorithm')
parser.add_argument('--sgd-weight-decay', default='0', type=str, help='weight decay value for the sgd algorithm')
parser.add_argument('--sgd-dampening', default='0', type=str, help='momentum value for the sgd algorithm')
parser.add_argument('--sgd-nesterov', default='False', type=str, help='enables nesterov momentum')


# adam 
parser.add_argument('--adam-lr', type=str, default='0.001', help='learning rate value for the adam algorithm')
parser.add_argument('--adam-beta1', type=str, default='0.9', help='beta1 value in the adam algorithm')
parser.add_argument('--adam-beta2', type=str, default='0.999', help='beta2 value in the adam algorithm')
parser.add_argument('--adam-eps', type=str, default='1e-8', help='eps value in the adam algorithm')
parser.add_argument('--adam-weight-decay', type=str, default='0', help='weight decay in the adam algoright')
parser.add_argument('--adam-amsgrad', type=str, default='False', help='use AMSgrad variance of the adam algorithm')

# rmsprop
parser.add_argument('--rmsprop-lr', type=str, default='0.01', help='learning rate value for the adam algorithm')
parser.add_argument('--rmsprop-momentum', type=str, default='0', help='momentum value in the rmsprop algorithm')
parser.add_argument('--rmsprop-alpha', type=str, default='0.99', help='alpha value in the rmsprop algoright')
parser.add_argument('--rmsprop-eps', type=str, default='1e-8', help='eps value in the rmsporp algorithm')
parser.add_argument('--rmsprop-centered', type=str, default='False', help='if True compute the centered rmsprop')
parser.add_argument('--rmsprop-weight-decay', type=str, default='0',  help='weight decay in the rmsprop algorithmt')

# adagrad
parser.add_argument('--adagrad-lr', type=str, default='0.001', help='learning rate value for the adagrad algorithm')
parser.add_argument('--adagrad-learning-decay', type=str, default='0.99', help='learning rate decay in the adagrad algoright')
parser.add_argument('--adagrad-weight-decay', type=str, default='0',  help='weight decay in the adagrad algorithmt')
parser.add_argument('--adagrad-initial-acumulator', type=str, default='0', help='initial acumulator value in the adagrad algorithm')

# parser.add_argument('-wd', '--weight-decay', default=5*1e-4, type=float
#                     help='weight decay (default: 5*1e-4)')                    


# parser.add_argument('-lr', '--original-learning-rate', default=0.01, type=float, metavar='LR',
#                     help='initial learning rate')

# Learnig rate decay method

parser.add_argument('--learning-method', default='constant', type=str,
                    help='select the training method and its parameters. acepted methods: constant, fixed interval, fixed percentage, proportional, exponential, dynamic, TAS, TAL, cossine')


# Learnig rate decay - Hyperparameters 

# fixed epochs
parser.add_argument('--fixed-epochs-milestgones', default="40 70 90", type=str,
                    help='percentage of total amount of epochs to decrease the learning rate')
parser.add_argument('--fixed-epochs-rate', default='0.1', type=str,
                    help='learning rate decay rate')

# fixed interval                    
parser.add_argument('--fixed-interval-period', default='30', type=str,
                    help='learning rate decay period')
parser.add_argument('--fixed-interval-rate', default='0.1', type=str,
                    help='learning rate decay rate')

# TAS
parser.add_argument('--tas-alpha', default='20', type=str,
                    help='learning rate decay rate')
parser.add_argument('--tas-beta', default='0.5', type=str,
                    help='learning rate decay rate')
parser.add_argument('--tas-gamma', default='1e-3', type=str,
                    help='learning rate decay rate')


# Misc
parser.add_argument('--executions', default='5', type=str, metavar='N',
                    help='Number of executions (default: 5)')
parser.add_argument('--print-freq', default=16, type=int, metavar='N',
                    help='print frequency (default: 16)')
parser.add_argument('--base-seed', default='1230', type=str,
                    help='seed for pseudo random number generator (default: 1230)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id for CUDA_VISIBLE_DEVICES')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dataset-dir', type=str, metavar='DATA',
                    help='output dir for logits extracted')
parser.add_argument('--results-dir', default='test', type=str,
                    help='folder to put the results')


args = parser.parse_args()
args_dict = vars(args)

general_parameters = ['architecture', 'dataset', 'epochs', 'executions',
                      'batch_size', 'base_seed', #'initial_learning_rate',
                      'test_set_split', 'validation_set_split', 'combine_datasets', 'do_validation_set', 'reduce_train_set']


parameters_to_ignore = ['print_freq', 'gpu_id', 'workers', 'dataset_dir',
                        'results_dir']

# TODO change how diferent types of models are handled
# if args.local_model is not None:
#     achitecture_parameters = ['local_model']
# else:
#      achitecture_parameters = ['remote_model']


full_experiment_dict = {}


for parameter_name, parameter_value in args_dict.items():
    if parameter_name not in parameters_to_ignore:
        print(parameter_name)
        n_elements = parameter_value.count(',') + 1

        if n_elements > 1:
            parameter_value = parameter_value.split(',')

        if isinstance(parameter_value, list):
            full_experiment_dict[parameter_name] = parameter_value
        else:
            full_experiment_dict[parameter_name] = [parameter_value]


general_parameters_dic = {key:value for key, value in
                          full_experiment_dict.items() if
                          key in general_parameters}



training_parameters_dic = {'adam':{key:value for key, value in
                                   full_experiment_dict.items() if
                                   key.count('adam_') == 1},
                           'adagrad':{key:value for key, value in
                                      full_experiment_dict.items() if
                                      key.count('adagrad_') == 1},
                            'sgd':{key:value for key, value in
                                      full_experiment_dict.items() if
                                      key.count('sgd') == 1},
                           'rmsprop':{key:value for key, value  in
                                      full_experiment_dict.items() if
                                      key.count('rmsprop_') == 1}}


learning_parameters_dic = {'tas': {key: value for key, value in
                                   full_experiment_dict.items() if
                                   key.count('tas_') == 1},
                           'constant': {key: value for key, value in
                                        full_experiment_dict.items() if
                                        key.count('constant_') == 1},
                           'fixed_epochs': {key: value for key, value in
                                            full_experiment_dict.items() if
                                            key.count('fixed_epochs_') == 1},
                           'fixed_interval': {key: value for key, value in
                                              full_experiment_dict.items() if
                                              key.count('fixed_interval_') == 1}}

print('GENERAL PARAMETERS\n',general_parameters_dic)
print('TRAINING PARAMETERS\n', training_parameters_dic)
print('LEARNING PARAMETERS\n', learning_parameters_dic)


aux_experiment_list = []

for training_method in full_experiment_dict['training_method']:
    partial_experiment_dic = {**training_parameters_dic[training_method],
                              **general_parameters_dic}
    partial_experiment_dic['training_method'] = [training_method]

    for learning_method in full_experiment_dict['learning_method']:
        experiment_dic = {**learning_parameters_dic[learning_method],
                          **partial_experiment_dic}

        experiment_dic['learning_method'] = [learning_method]
        aux_experiment_list.append(experiment_dic)


full_experiments_list = []

for aux_experiment in aux_experiment_list:
    aux_experiment_values = list(aux_experiment.values())
    experiment_keys = list(aux_experiment.keys())
    for experiment_values in itertools.product(*aux_experiment_values):

        full_experiments_list.append({})
        full_experiments_list[-1]['parameters'] = dict(zip(experiment_keys,
                                                          experiment_values))

        n_execution = int(args.executions)

        full_experiments_list[-1]['raw_results'] = {'train_loss': [[] for _ in range(n_execution)],
                                                    'train_acc1': [[] for _ in range(n_execution)],
                                                    'val_acc1': [[] for _ in range(n_execution)],
                                                    'val_loss': [[] for _ in range(n_execution)],
                                                    'learning_rate': [[] for _ in range(n_execution)],
                                                    'val_acc1_list': [[] for _ in range(n_execution)],
                                                    'val_loss_list': [[] for _ in range(n_execution)],
                                                    'train_acc1_list': [[] for _ in range(n_execution)],
                                                    'train_loss_list': [[] for _ in range(n_execution)]}

print(len(full_experiments_list))
time.sleep(5)
# cuda device to be used...
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
# TODO seed


def execute(experiment):
    print('EXECUTE')
    print(experiment['parameters'])
    time.sleep(3)
    parameters = experiment['parameters'].copy()
    raw_results = experiment['raw_results']
    args.experiment_path = os.path.join("artifacts", parameters['dataset'], parameters['architecture'])


    ######################################
    # Initial configuration...
    ######################################

    float_parameters = [
        'test_set_split',
        'reduce_train_set',
        'validation_set_split',
        'sgd_lr',
        'sgd_momentum',
        'sgd_weight_decay',
        'sgd_dampening',
        'adam_lr',
        'adam_beta1',
        'adam_beta2',
        'adam_eps',
        'adam_weight_decay',
        'rmsprop_lr',
        'rmsprop_momentum',
        'rmsprop_alpha',
        'rmsprop_eps',
        'rmsprop_weight_decay',
        'adagrad_lr',
        'adagrad_learning_decay',
        'adagrad_weight_decay',
        'adagrad_initial_acumulator',
        'tas_alpha',
        'tas_beta',
        'tas_gamma']

    int_parameters = [
        'epochs',
        'batch_size',
        'executions',
        'base_seed']

    bool_parameters = [
        'do_validation_set',
        'combine_datasets',
        'sgd_nesterov',
        'adam_amsgrad',
        'rmsprop_centered']


    for float_parameter in float_parameters:
        if float_parameter in parameters.keys():
            parameters[float_parameter] = float(parameters[float_parameter])

    for int_parameter in int_parameters:
        if int_parameter in parameters.keys():
            parameters[int_parameter] = int(parameters[int_parameter])


    for bool_parameter in bool_parameters:
        if bool_parameter in parameters.keys():
            parameters[bool_parameter] = parameters[bool_parameter] == 'True'

    # Using seeds...
    random.seed(parameters['base_seed'])
    numpy.random.seed(parameters['base_seed'])
    torch.manual_seed(parameters['base_seed'])
    torch.cuda.manual_seed(parameters['base_seed'])
    args.execution_seed = parameters['base_seed'] + args.execution
    print("EXECUTION SEED:", args.execution_seed)

    print(args.dataset)
    # Configuring args and dataset...
    if args.dataset == "mnist":
        args.number_of_dataset_classes = 10
        args.number_of_model_classes = args.number_of_dataset_classes
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        train_transform = transforms.Compose(
            [transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
        dataset_path = args.dataset_dir if args.dataset_dir else "datasets/mnist"
        train_set = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True, transform=inference_transform)
    elif args.dataset == "cifar10":
        args.number_of_dataset_classes = 10
        args.number_of_model_classes = args.number_of_dataset_classes
        normalize = transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
        dataset_path = args.dataset_dir if args.dataset_dir else "datasets/cifar10"
        train_set = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=inference_transform)
    elif args.dataset == "cifar100":
        args.number_of_dataset_classes = 100
        args.number_of_model_classes = args.number_of_dataset_classes
        normalize = transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
        dataset_path = args.dataset_dir if args.dataset_dir else "datasets/cifar100"
        train_set = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=inference_transform)
    else:
        args.number_of_dataset_classes = 1000
        args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 1000
        if args.arch.startswith('inception'):
            size = (299, 299)
        else:
            size = (224, 256)
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(size[0]),  # 224 , 299
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose(
            [transforms.Resize(size[1]),  # 256
             transforms.CenterCrop(size[0]),  # 224 , 299
             transforms.ToTensor(), normalize])
        dataset_path = args.dataset_dir if args.dataset_dir else "/mnt/ssd/imagenet_scripts/2012/images"
        train_path = os.path.join(dataset_path, 'train')
        val_path = os.path.join(dataset_path, 'val')
        train_set = ImageFolder(train_path, transform=train_transform)
        test_set = ImageFolder(val_path, transform=inference_transform)

    # Preparing paths...
    # TODO make execution path an input
    args.execution_path = 'results'
    if not os.path.exists(args.execution_path):
        os.makedirs(args.execution_path)


    ######################################
    # Preparing data...
    ######################################
    if parameters['combine_datasets']:
        complete_dataset = torch.utils.data.ConcatDataset((train_set, test_set))
        train_set, test_set = torch.utils.data.random_split(complete_dataset,
            [round((1 - parameters['test_set_split'])*len(complete_dataset)),
            round(parameters['test_set_split']*len(complete_dataset))])
    if parameters['do_validation_set']:
        train_set, validation_set = torch.utils.data.random_split(train_set,
            [round((1 - parameters['validation_set_split'])*len(complete_dataset)),
            round(parameters['validation_set_split']*len(complete_dataset))])


    if parameters['reduce_train_set'] != 1.0:
        train_set, _ = torch.utils.data.random_split(train_set, [round(parameters['reduce_train_set'] * len(train_set)), round((1 - parameters['reduce_train_set']) * len(train_set))])
        test_set, _ = torch.utils.data.random_split(test_set, [round(parameters['reduce_train_set'] * len(test_set)), round((1 - parameters['reduce_train_set']) * len(test_set))])


    # TODO make shuffle a general parameter
    train_loader = DataLoader(train_set,
                              batch_size=parameters['batch_size'],
                              num_workers=args.workers,
                              shuffle=True)

    test_loader = DataLoader(test_set,
                            batch_size=parameters['batch_size'],
                            num_workers=args.workers,
                            shuffle=True)

    print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("TRAINSET LOADER SIZE: ====>>>> ", len(train_loader.sampler))
    print("TESTSET LOADER SIZE: ====>>>> ", len(test_loader.sampler))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # Dataset created...
    print("\nDATASET:", args.dataset)

    # create model
    torch.manual_seed(args.execution_seed)
    torch.cuda.manual_seed(args.execution_seed)
    print("=> creating model '{}'".format(parameters['architecture']))
    # model = create_model()
    model = models.__dict__[parameters['architecture']](num_classes=args.number_of_model_classes)
    model.cuda()
    print("\nMODEL:", model)
    torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed(args.base_seed)
    #########################################
    # Training...
    #########################################

    # define loss function (criterion)...
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer..

    if parameters['training_method'] == 'sgd':

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=parameters['sgd_lr'],
                                    momentum=parameters['sgd_momentum'],
                                    weight_decay=parameters['sgd_weight_decay'],
                                    nesterov=parameters['sgd_nesterov'])

    elif parameters['training_method'] == 'adam':
        print('****************AMSGRAD*************')
        print(parameters['adam_amsgrad'])
        time.sleep(3)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=parameters['adam_lr'],
                                     betas=[parameters['adam_beta1'], parameters['adam_beta2']],
                                     eps=parameters['adam_eps'],
                                     weight_decay=parameters['adam_weight_decay'],
                                     amsgrad=parameters['adam_amsgrad'])

    elif parameters['training_method'] == 'adagrad':

        optimizer = torch.optim.Adagrad(model.parameters(),
                                     lr=parameters['adagrad_lr'],
                                     lr_decay=parameters['adagrad_learning_decay'],
                                     weight_decay=parameters['adagrad_weight_decay'])

    elif parameters['training_method'] == 'rmsprop':

        optimizer = torch.optim.RMSprop(model.parameters(),
                                     lr=parameters['rmsprop_lr'],
                                     momentum=parameters['rmsprop_momentum'],
                                     alpha=parameters['rmsprop_alpha'],
                                     eps=parameters['rmsprop_eps'],
                                     centered=parameters['rmsprop_centered'],
                                     weight_decay=parameters['rmsprop_weight_decay'])

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # , weight_decay=5e-4)

    # define scheduler...
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.2, verbose=True,
    #                         
    #                               threshold=0.05, threshold_mode='rel')

    print(parameters)
    if parameters['learning_method'] == 'tas':
        alpha = parameters['tas_alpha']
        beta = parameters['tas_beta']
        gamma = parameters['tas_gamma']
        our_lambda = lambda epoch: (1 - gamma)/(1 + math.exp(alpha*(epoch/parameters['epochs']-beta))) + gamma
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=our_lambda)

    if parameters['learning_method'] == 'fixed_interval':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=parameters['fixed_interval_rate'],
                                                    gamma=parameters['fixed_interval_period'])

    if parameters['learning_method'] == 'fixed_epochs':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=parameters['fixed_epochs_milestones'],
                                                         gamma=parameters['fixed_epochs_rate'])

    if parameters['learning_method'] == 'constant':
        scheduler = None
    # model.initialize_parameters() ####### It works for AlexNet_, LeNet and VGG...
    # initialize_parameters(model)

    print("\n################ TRAINING ################")
    best_model_file_path = os.path.join(args.execution_path, 'best_model.pth.tar')
    best_train_acc1, best_val_acc1, final_train_loss = \
        train_val(parameters, raw_results, train_loader, test_loader, model, criterion, optimizer,
                  scheduler, best_model_file_path)

    # save to json file

    return best_train_acc1, best_val_acc1, final_train_loss, raw_results


def train_val(parameters, raw_results, train_loader, test_loader, model, criterion, optimizer, scheduler,
              best_model_file_path):

    best_model_train_acc1 = -1
    best_model_val_acc1 = -1
    best_model_train_loss = None


    # for epoch in range(start_epoch, end_epoch + 1):
    for epoch in range(1, parameters['epochs'] + 1):
        print("\n######## EPOCH:", epoch, "OF", parameters['epochs'], "########")

        # Adjusting learning rate (if not using reduce on plateau)...
        if scheduler is not None:
            scheduler.step()

        # Print current learning rate...

        for param_group in optimizer.param_groups:
            print("\nLEARNING RATE:\t", param_group["lr"])

        train_acc1, train_loss, train_acc_list, train_loss_list = train(train_loader, model, criterion, optimizer, epoch)
        val_acc1, val_loss, val_acc_list, val_loss_list  = validate(test_loader, model, criterion, epoch)

        # Saving raw results...
        raw_results['train_acc1'][args.execution - 1].append(round(train_acc1*1000)/1000)
        raw_results['val_acc1'][args.execution - 1].append(round(val_acc1*1000)/1000)
        raw_results['train_acc1_list'][args.execution - 1].extend(train_acc_list)
        raw_results['val_acc1_list'][args.execution - 1].extend(val_acc_list)
        raw_results['train_loss'][args.execution - 1].append(round(train_loss*1000)/1000)
        raw_results['val_loss'][args.execution - 1].append(round(val_loss*1000)/1000)
        raw_results['train_loss_list'][args.execution - 1].extend(train_loss_list)
        raw_results['val_loss_list'][args.execution - 1].extend(val_loss_list)
        raw_results['learning_rate'][args.execution - 1].append(round(param_group["lr"]*1000000)/1000000)

        # if is best...
        if val_acc1 > best_model_val_acc1:

            best_model_train_acc1 = train_acc1
            best_model_val_acc1 = val_acc1
            best_model_train_loss = train_loss


            print("!+NEW BEST+ {0:.3f} IN EPOCH {1}!!! SAVING... {2}\n".format(val_acc1, epoch, best_model_file_path))
            full_state = {'epoch': epoch, 'arch': parameters['architecture'], 'model_state_dict': model.state_dict(), 'best_val_acc1': best_model_val_acc1}
            torch.save(full_state, best_model_file_path)

        print('!$$$$ BEST: {0:.3f}\n'.format(best_model_val_acc1))

        # Adjusting learning rate (if using reduce on plateau)...
        #### scheduler.step(val_acc1)
        #scheduler.step(train_loss)

    return best_model_train_acc1, best_model_val_acc1, best_model_train_loss


def train(train_loader, model, criterion, optimizer, epoch):
    # Meters...
    train_loss = tnt.meter.AverageValueMeter()
    train_acc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    train_conf = tnt.meter.ConfusionMeter(args.number_of_model_classes, normalized=True)

    # switch to train mode
    model.train()

    # Start timer...
    train_batch_start_time = time.time()

    acc_list = []
    loss_list = []
    for batch_index, (input_tensor, target_tensor) in enumerate(train_loader):
        batch_index += 1

        # measure data loading time
        train_data_time = time.time() - train_batch_start_time

        # moving to GPU...
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda(non_blocking=True)

        # compute output
        output_tensor = model(input_tensor)

        # compute loss

        loss = criterion(output_tensor, target_tensor)

        # accumulate metrics over epoch
        train_loss.add(loss.item())
        train_acc.add(output_tensor.detach(), target_tensor.detach())
        train_conf.add(output_tensor.detach(), target_tensor.detach())

        # zero grads, compute gradients and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        train_batch_time = time.time() - train_batch_start_time

        if batch_index % args.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Data {train_data_time:.6f}\t'
                  'Time {train_batch_time:.6f}\t'
                  'Loss {loss:.4f}\t'
                  'Acc1 {acc1_meter:.2f}\t'
                  'Acc5 {acc5_meter:.2f}'
                  .format(epoch, batch_index, len(train_loader),
                          train_data_time=train_data_time,
                          train_batch_time=train_batch_time,
                          loss=train_loss.value()[0],
                          acc1_meter=train_acc.value()[0],
                          acc5_meter=train_acc.value()[1],
                          )
                  )
        acc_list.append(round(train_acc.value()[0]*1000)/1000)
        loss_list.append(round(train_acc.value()[0]*1000)/1000)
        # Restart timer...
        train_batch_start_time = time.time()

    print("\nCONFUSION:\n", train_conf.value())
    print('\n#### TRAIN: {acc1:.3f}\n\n'.format(acc1=train_acc.value()[0]))


    return train_acc.value()[0], train_loss.value()[0], acc_list, loss_list


def validate(test_loader, model, criterion, epoch):
    # Meters...
    val_loss = tnt.meter.AverageValueMeter()
    val_acc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    val_conf = tnt.meter.ConfusionMeter(args.number_of_model_classes, normalized=True)

    # switch to evaluate mode
    model.eval()

    #correct = 0
    #total = 0

    # Start timer...
    val_batch_start_time = time.time()

    acc_list = []
    loss_list = []

    with torch.no_grad():

        for batch_index, (input_tensor, target_tensor) in enumerate(test_loader):
            batch_index += 1

            # measure data loading time
            val_data_time = time.time()-val_batch_start_time

            """
            input_tensor = torch.autograd.Variable(input_tensor, volatile=True)
            target_tensor = target_tensor.cuda(async=True)
            target_tensor = torch.autograd.Variable(target_tensor, volatile=True)
            """

            # moving to GPU...
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda(non_blocking=True)

            # compute output
            output_tensor = model(input_tensor)
            # compute loss
            loss = criterion(output_tensor, target_tensor)

            # accumulate metrics over epoch
            val_acc.add(output_tensor.detach(), target_tensor.detach())
            val_loss.add(loss.item())
            val_conf.add(output_tensor.detach(), target_tensor.detach())

            # measure elapsed time
            val_batch_time = time.time()-val_batch_start_time

            if batch_index % args.print_freq == 0:
                print('Valid Epoch: [{0}][{1}/{2}]\t'
                      'Data {val_data_time:.6f}\t'
                      'Time {val_batch_time:.6f}\t'
                      'Acc1 {acc1_meter:.2f}\t'
                      'Acc5 {acc5_meter:.2f}'
                      .format(epoch, batch_index, len(test_loader),
                              val_data_time=val_data_time,
                              val_batch_time=val_batch_time,
                              acc1_meter=val_acc.value()[0],
                              acc5_meter=val_acc.value()[1],
                              )
                      )

            acc_list.append(round(val_acc.value()[0]*1000)/1000)
            loss_list.append(round(val_loss.value()[0]*1000)/1000)

            # Restart timer...
            val_batch_start_time = time.time()

    print("\nCONFUSION:\n", val_conf.value())
    print('\n#### VALID: {acc1:.3f}\n'.format(acc1=val_acc.value()[0]))

    return val_acc.value()[0], val_loss.value()[0], acc_list, loss_list

# TODO what is this?
def worker_init(worker_id):
    random.seed(args.base_seed)

def main():

    overall_stats = {}

    for i_experiment, experiment in enumerate(full_experiments_list):

        print("\n\n")
        print("****************************************************************")
        print("EXPERIMENT:", experiment['parameters'])
        print("****************************************************************\n")
        time.sleep(5)
        # execution_results = {}
        experiment_stats = pd.DataFrame()

        # args.experiment_path = os.path.join("artifacts", args.dataset, args.arch, experiment)
        # print("PATH:", args.experiment_path)



        for args.execution in range(1, int(args.executions) + 1):

            # args.execution = execution
            execution_results = {}

            print("\n################ EXECUTION:", args.execution, "OF", args.executions, "################")

            # execute and get results and statistics...
            (execution_results["TRAIN [ACC1]"], execution_results["VAL [ACC1]"],
            execution_results["TRAIN LOSS"], experiment['raw_results']) = execute(experiment)
            print(experiment['raw_results'])
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            # appending results...
            experiment_stats = experiment_stats.append(execution_results, ignore_index=True)
            experiment_stats = experiment_stats[["TRAIN [ACC1]", "VAL [ACC1]", "TRAIN LOSS"]]


        results_directory = os.path.join('results',
                                         experiment['parameters']['dataset'],
                                         experiment['parameters']['architecture'],
                                         args.results_dir)

        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        csv.register_dialect('unixpwd', delimiter=',', quoting=csv.QUOTE_NONE)

        file_name = experiment['parameters']['training_method'] + '_' + experiment['parameters']['learning_method'] + '-' + str(datetime.now())
        # for key, value in experiment['parameters'].items():
        #         file_name = file_name + key + ':' + str(value) + ','            
        # file_name = file_name[:-1]

        file_path = os.path.join(results_directory, file_name + '.txt')

        with open(file_path, 'w+', newline='') as csvfile:

            writer = csv.writer(csvfile, dialect='unixpwd', delimiter=',')
            csvfile.write('parameters\n')

            print(experiment['parameters'])
            time.sleep(3)
            for key, value in experiment['parameters'].items():
                csvfile.write(key + ':' + str(value) + '\n')

            csvfile.write('\n')

            for key, value in experiment['raw_results'].items():
                if 'list' not in key:
                    csvfile.write(key + '\n')
                    for i_execution, execution in enumerate(value):
                        writer.writerow(execution)
                    csvfile.write('\n')
            # for execution in range(args.executions):
            #     for key in experiment['raw_results']:
            #         writer.
            #         writer.writerow(experiment['raw_results'][key][execution])

        # overall_stats[i_experiment] = experiment_stats


    #print("\n\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n", "OVERALL STATISTICS", "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    #for key in overall_stats:
    #    print("\n", key.upper())
    #    print("\n", overall_stats[key].transpose())
    #    print("\n", overall_stats[key].describe())
        # print("\n", overall_stats[key].describe().loc[['mean', 'std']])

    #print("\n\n\n")



if __name__ == '__main__':
    main()
