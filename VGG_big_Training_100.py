from __future__ import print_function, division
import argparse
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime
import copy
import time

parser = argparse.ArgumentParser(description='VGG16 CIFAR100 learning')

parser.add_argument('--batchSize', '-b', default=32, type=float, metavar='N',
                    help='minibatch size (default: 32)')
parser.add_argument('--regularization', '-r', default=0.2e-4, type=float, metavar='N',
                    help='')
parser.add_argument('--nEpochs', '-e', default=25, type=int, metavar='N',
                    help='')
parser.add_argument('--nLearningDecay', '-l', default=7, type=int, metavar='N',
                    help='')

args = parser.parse_args()


print('Execution started: ')
print(datetime.now().time())
print('Batch size: %d'%(args.batchSize))
print('Weight decay: %f'%(args.regularization))
print('Number of epochs: %d'%(args.nEpochs))
print('Epochs between LR decay: %d'%(args.nLearningDecay))

minibatch_size = args.batchSize

file = open('transfer_results_batch%d.txt'%(minibatch_size), 'w')

# load CIFAR10
transform = transforms.Compose(
    [transforms.Scale(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size, shuffle=False, num_workers=4)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

use_gpu = torch.cuda.is_available()


def exp_lr_scheduler(optimizer, epoch, init_lr=0.00005 * minibatch_size, lr_decay_epoch=args.nLearningDecay):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1)**(epoch // lr_decay_epoch)

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

#load the big network
model_big = models.vgg16(pretrained=True)

model_big.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 100),
        )

# use if only classifier is trained
#for param in model_big.parameters():
#    param.requires_grad = False
#for param in model_big.classifier.parameters():
#    param.requires_grad = True

if use_gpu:
    model_big = model_big.cuda()

#Training

optimizer = optim.SGD(model_big.parameters(), lr=0.00005 * minibatch_size, momentum=0.9)
#optimizer = optim.Adam(model_big.parameters(), lr = 0.00005 * minibatch_size, weight_decay=args.regularization)

lr_scheduler = exp_lr_scheduler

best_model = model_big
best_acc = 0.0
best_loss = 0.0

hard_criterion = nn.CrossEntropyLoss()

num_epochs = args.nEpochs

since = time.time()

for epoch in range(num_epochs):
    #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    #print('-' * 10)
    optimizer = lr_scheduler(optimizer, epoch)

    model_big.train(True)  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), \
                             Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model_big(inputs)

        _, preds = torch.max(outputs.data, 1)

        loss = hard_criterion(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(trainset)
    epoch_acc = running_corrects / len(trainset)

    print('Epoch {}/{}, Training: Loss = {:.4f} Acc = {:.4f}'.format(epoch, num_epochs - 1,epoch_loss, epoch_acc))

    model_big.train(False)  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0


    model_big.train(False)

    for i, data in enumerate(testloader):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), \
                             Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model_big(inputs)

        _, preds = torch.max(outputs.data, 1)

        loss = hard_criterion(outputs, labels)

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(testset)
    epoch_acc = running_corrects / len(testset)

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = copy.deepcopy(model_big)

    if epoch_loss < best_loss or epoch == 0:
        best_loss = epoch_loss

    print('Epoch {}/{}, Testing: Loss = {:.4f} Acc = {:.4f}'.format(epoch, num_epochs - 1, epoch_loss, epoch_acc))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

torch.save(best_model.state_dict(), "modelBigStateDict_CIFAR100")