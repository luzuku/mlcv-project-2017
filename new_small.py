from __future__ import print_function, division
import argparse
import torch
import torchvision
import torchvision.models as models

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime
import copy
import time

from helper import load_cifa10

hyperparams = {}
hyperparams['batch_size'] = 32
hyperparams['init_lr'] = 0.001 * hyperparams['batch_size']
hyperparams['lr_decay_epoch'] = 18
hyperparams['epochs'] = 25
hyperparams['weight_decay'] = 0.2e-4
hyperparams['cfg'] = [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']



data_path = load_cifa10(path='../data', batch_size=hyperparams['batch_size'])

def VGG_small(hyperparams):
    cfg = hyperparams['cfg']
    model_small = models.vgg.VGG(models.vgg.make_layers(cfg))
    model_small.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                           nn.ReLU(True),
                                          #nn.Dropout(),
                                          #nn.Linear(4096, 4096),
                                          #nn.ReLU(True),
                                          #nn.Dropout(),
                                           nn.Linear(4096, 10))
    return model_small

def train(model, hyperparams):
    return None
