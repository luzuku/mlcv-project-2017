

def load_cifa10(path='../data', minibatch_size=32):
    '''Load cifar 10 '''
    transform = transforms.Compose([transforms.Scale(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=minibatch_size,
                                              shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size,
                                             shuffle=False, num_workers=4)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    return trainset, trainloader, testset, testloader, classes
