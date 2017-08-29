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
import time
import sys

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Transfer learning')

parser.add_argument('--softFactor', '-sf', default=1, type=float, metavar='N',
                    help='soft factor (default: 1)')
parser.add_argument('--logitFactor', '-lf', default=1, type=float, metavar='N',
                    help='logit factor (default: 1)')
parser.add_argument('--hardFactor', '-hf', default=0.2, type=float, metavar='N',
                    help='hard factor (default: .2)')
parser.add_argument('--temperature', '-T', default = 1.0, type=float, metavar='N',
                    help='temperature (default: 1)')
parser.add_argument('--batchSize', '-b', default=32, type=float, metavar='N',
                    help='minibatch size (default: 32)')
parser.add_argument('--regularization', '-r', default=0.5e-4, type=float, metavar='N',
                    help='')
parser.add_argument('--nEpochs', '-e', default=25, type=int, metavar='N',
                    help='')
parser.add_argument('--nLearningDecay', '-l', default=7, type=int, metavar='N',
                    help='')

args = parser.parse_args()



#ActivationSaver class used to save intermediate activations of nets during forward propagation
class ActivationSaver:

    def __init__(self, modules):
        self.activations = {}

        def output_saver_hook(module, input, output):
            self.activations[module] = output

        for module in modules:
            module.register_forward_hook(output_saver_hook)
            

# indices has shape [[feature indices], [classifier indices]]            
def intermediateModulesFromIndices(model, indices):
    result = []
    for i in indices[0]:
        result.append(model.features[i])
    for i in indices[1]:
        result.append(model.classifier[i])
    return result



def main(small_intermediate_indices, big_intermediate_indices, batch_size, hard_factor = .2, logit_factor = 1, soft_factor = 1, 
         temperature = 1.0, weight_decay = .5e-4, num_epochs = 25, nLearningDecay = 7, testBigModel = False, filename = None):
    
    if filename != None:
        file = open(filename, 'w')
        sys.stdout = file
        
    print('Execution started: ')
    print(datetime.now().time())
    print('Batch size: %d'%(batch_size))
    print('Hard factor: %f'%(hard_factor))
    print('Logit factor: %f'%(logit_factor))
    print('Temperature: %f'%(temperature))
    print('Soft factor: %f'%(soft_factor))
    print('Weight decay: %f'%(weight_decay))
    print('Number of epochs: %d'%(num_epochs))
    print('Epochs between LR decay: %d'%(nLearningDecay))
    
    use_gpu = torch.cuda.is_available()

    # load CIFAR10
    transform = transforms.Compose(
        [transforms.Scale(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    #load the big network
    model_big = models.vgg16()

    model_big.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 10),
            )

    model_big.load_state_dict(torch.load("modelBigStateDict250817"))

    #gradients not requiered
    for param in model_big.parameters():
        param.requires_grad = False

    model_big.train(False)

    if use_gpu:
        model_big = model_big.cuda()

    print("Big model initialized")

    if testBigModel:
        running_corrects = 0
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

            # statistics
            running_corrects += torch.sum(preds == labels.data)

        big_acc = running_corrects / len(testset)

        print('Big Model Accuracy: {:.4f}'.format(big_acc))


    # define the small network
    cfg = [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']

    model_small = models.vgg.VGG(models.vgg.make_layers(cfg))
    model_small.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                #nn.Dropout(),
                #nn.Linear(4096, 4096),
                #nn.ReLU(True),
                #nn.Dropout(),
                nn.Linear(4096, 10),
            )

    if use_gpu:
        model_small = model_small.cuda()

    print("Small model initialized")


    #setting up the ActivationSavers

    model_big_intermediate_modules = intermediateModulesFromIndices(model_big, big_intermediate_indices)

    model_small_intermediate_modules = intermediateModulesFromIndices(model_small, small_intermediate_indices)

    activations_big = ActivationSaver(model_big_intermediate_modules)
    activations_small = ActivationSaver(model_small_intermediate_modules)

    nTransferLayers = len(model_big_intermediate_modules)

    
    #Training
    
    # used to adjust the learning rate
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.0001 * batch_size, lr_decay_epoch=nLearningDecay):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.1**(epoch // lr_decay_epoch))

        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer
    
    #optimizer = optim.SGD(model_small.parameters(), lr=0.0001 * batch_size, momentum=0.9)
    optimizer = optim.Adam(model_small.parameters(), lr = 0.0001 * batch_size, weight_decay=args.regularization)

    lr_scheduler = exp_lr_scheduler

    best_model = model_small
    best_acc = 0.0

    soft_criterion = nn.MSELoss()
    hard_criterion = nn.CrossEntropyLoss()

    logSoftmax1 = nn.Softmax()
    logSoftmax2 = nn.Softmax()
    crossEntropy = nn.KLDivLoss()

    def logit_criterion(input, target):
        return (temperature ** 2) * crossEntropy(logSoftmax1(input / temperature), logSoftmax2(target / temperature))

    since = time.time()

    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        optimizer = lr_scheduler(optimizer, epoch)
        model_small.train(True)  # Set model to training mode

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
            small_outputs = model_small(inputs)
            if soft_factor > 0 or logit_factor >0:
                big_outputs = Variable(model_big(inputs).data)

            _, preds = torch.max(small_outputs.data, 1)

            #hard loss
            loss = hard_factor * hard_criterion(small_outputs, labels)
            
            #add logit-layer loss
            loss = loss + logit_factor * logit_criterion(small_outputs, big_outputs)

            # add the contributions of the intermediate layers
            if soft_factor > 0:
                for j, small_act in enumerate(model_small_intermediate_modules):
                    big_act = model_big_intermediate_modules[j]

                    loss = loss + soft_factor/nTransferLayers * soft_criterion(activations_small.activations[small_act],
                                                               Variable(activations_big.activations[big_act].data))

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

            #if i % 10 == 9:  # print every 100 mini-batches
            #    print('%d minibatches processed' %
            #          (i + 1))
            #    print('%f \r\n' % (running_loss / (i+1)))

        epoch_loss = running_loss / len(trainset)
        epoch_acc = running_corrects / len(trainset)

        print('Epoch {}/{}, Training: Loss = {:.4f} Acc = {:.4f}'.format(epoch, num_epochs - 1,epoch_loss, epoch_acc))

        model_small.train(False)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

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
            outputs = model_small(inputs)
            if soft_factor > 0 or logit_factor > 0:
                model_big(inputs)

            _, preds = torch.max(outputs.data, 1)
            
            #hard loss
            loss = hard_factor * hard_criterion(small_outputs, labels)
            
            #add logit-layer loss
            loss = loss + logit_factor * logit_criterion(small_outputs, big_outputs)

            # add the contributions of the intermediate layers
            if soft_factor > 0:
                for j, small_act in enumerate(model_small_intermediate_modules):
                    big_act = model_big_intermediate_modules[j]

                    loss = loss + soft_factor/nTransferLayers * soft_criterion(activations_small.activations[small_act],
                                                               Variable(activations_big.activations[big_act].data))

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

            #if i % 100 == 99:  # print every 100 mini-batches
            #    print('%d minibatches processed' %
            #          (i + 1))
            #    print('%f \r\n' % (running_loss / i))
            # running_loss = 0.0

        epoch_loss = running_loss / len(testset)
        epoch_acc = running_corrects / len(testset)

        print('Epoch {}/{}, Testing: Loss = {:.4f} Acc = {:.4f}'.format(epoch, num_epochs - 1, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    if write_to_file:
        file.close()
    
    
if __name__ == "__main__":
    
    small_intermediate_indices = [[],[]]
    big_intermediate_indices = [[],[]]

    main(small_intermediate_indices, big_intermediate_indices, args.batchSize, args.hardFactor, args.logitFactor, args.softFactor,
         args.temperature, args.regularization, args.nEpochs, args.nLearningDecay, False, False)