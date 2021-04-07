import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
batch_size = 10000

cifar_data = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())

cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=transforms.ToTensor())

cifar_train, cifar_val = torch.utils.data.random_split(cifar_data,[40000,10000],generator=torch.Generator().manual_seed(420))


train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)

validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False)

test_loader = DataLoader(cifar_test, batch_size= batch_size, shuffle=False)

