import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
batch_size = 10000

batch_size = 1000

cifar_train = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
<<<<<<< HEAD
cifar_validation = datasets.CIFAR10("./", train=False, download=True, transform=transforms.ToTensor())
validation_loader = DataLoader(cifar_validation, batch_size=batch_size, shuffle=False)

=======
train_loader, validation_loader = torch.utils.data.random_split(train_loader,[4,1])

cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)
>>>>>>> 8d6370750f928bcbbac7c7002248a693f7ec4d66
