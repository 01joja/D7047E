from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 1000

cifar_train = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
cifar_validation = datasets.CIFAR10("./", train=False, download=True, transform=transforms.ToTensor())
validation_loader = DataLoader(cifar_validation, batch_size=batch_size, shuffle=False)

