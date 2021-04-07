import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

batch_size = 10000
epochs = 15

cifar_train = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)

cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)


network = nn.Sequential(
    
    nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=3),

    nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),

    nn.Linear(320, 10),
    nn.Softmax()
)
networkcopy = copy.deepcopy(network)
optimizer = optim.SGD(network.parameters(), lr = 0.0001)
loss_function = nn.CrossEntropyLoss()
traininglosses = []
validationlosses = []

for epoch in range(epochs):
    new_validationloss = 0
    
    for train_nr, (images, labels) in enumerate(train_loader):
        prediction = network(images)
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(
            '\rEpoch {} [{}/{}] - Loss: {}'.format(
                epoch+1, train_nr+1, len(train_loader), loss
            ),
            end=''
        )
        
    images, labels = next(iter(validation_loader))
    prediction = network(images)
    new_validationloss += loss_function(prediction, labels)
        
    if  new_validationloss < validationloss:
        validationloss = new_validationloss
        networkcopy = copy.deepcopy(network)

            
    traininglosses.append(loss)
    validationlosses.append(validationloss)

