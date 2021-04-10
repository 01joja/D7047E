import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import copy
import matplotlib.pyplot as plt
import numpy as np


batch_size = 1000
epochs = 25

cifar_data = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())
cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=transforms.ToTensor())

cifar_train, cifar_val = torch.utils.data.random_split(cifar_data,[40000,10000])


train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(cifar_test, batch_size= batch_size, shuffle=False)

dataiter = iter(train_loader)
images, labels = dataiter.next()

plt.imshow(np.transpose(torchvision.utils.make_grid(images).numpy(),(1,2,0)))
plt.show()


network = nn.Sequential(
    
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,padding=1),
    nn.Tanh(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1),
    nn.Tanh(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),

    nn.Linear(8192, 10)
)
networkcopy = copy.deepcopy(network)
optimizer = optim.Adam(network.parameters(), lr = 0.0001)
loss_function = nn.CrossEntropyLoss()
traininglosses = []
validationlosses = []
firstRun = True

for epoch in range(epochs):
    new_trainingloss = 0
    
    for train_nr, (images, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        prediction = network(images)
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        print(
            '\rEpoch {} [{}/{}] - Loss: {}'.format(
                epoch+1, train_nr+1, len(train_loader), loss
            ),
            end=''
        )
        new_trainingloss += loss

    traininglosses.append(new_trainingloss)
    new_validationloss = 0
    
    for val_nr, (images, labels) in enumerate(validation_loader):   
        prediction = network(images)
        new_validationloss += loss_function(prediction, labels)

    if firstRun:
        validationloss = new_validationloss
        firstRun = False
    elif  new_validationloss < validationloss:
        validationloss = new_validationloss
        networkcopy = copy.deepcopy(network)

    validationlosses.append(validationloss)
corr = 0

for index, (image, label) in enumerate(test_loader):
    guess = torch.argmax(network(image), dim=-1)
    corr += (guess == label).sum()
print("\n","Result on test:", corr/10000)
print(traininglosses)