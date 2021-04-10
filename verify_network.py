import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import pickle

batch_size = 10000

cifar_data = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())
cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=transforms.ToTensor())

cifar_train, cifar_val = torch.utils.data.random_split(cifar_data,[40000,10000],generator=torch.Generator().manual_seed(420))

train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(cifar_test, batch_size= batch_size, shuffle=False)


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))




