import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from .util import readmts_uci_har, transform_labels
def getDataset(dataset):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        ])

    if(dataset == 'CIFAR10'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        inputs=3

    elif(dataset == 'CIFAR100'):
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
        inputs = 3
        
    elif(dataset == 'MNIST'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        inputs = 1

    elif (dataset == 'UCI'):
        file_name = 'datasets/UCI_HAR_Dataset'
        x_train, y_train, x_test, y_test = readmts_uci_har(file_name)
        data = np.concatenate((x_train, x_test), axis=0)
        label = np.concatenate((y_train, y_test), axis=0)
        N = data.shape[0]
        ind = int(N * 0.9)
        x_train = data[:ind]
        y_train = label[:ind]
        x_test = data[ind:]
        y_test = label[ind:]
        y_train, y_test = transform_labels(y_train, y_test)

        x_train = x_train.swapaxes(1, 2)
        x_test = x_test.swapaxes(1, 2)

        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train)
        print(x_train.size(), y_train.size())
        trainset = torch.utils.data.TensorDataset(x_train, y_train)

        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test)
        print(x_test.size(), y_test.size())
        testset = torch.utils.data.TensorDataset(x_test, y_test)

        num_classes = 6

        inputs = 9

    return trainset, testset, inputs, num_classes

def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    print(len(train_loader))
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    print(len(valid_loader))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
        num_workers=num_workers)
    print(len(test_loader))

    return train_loader, valid_loader, test_loader