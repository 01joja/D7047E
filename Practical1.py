import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

batch_size = 10000
epochs = 50

cifar_data = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())
cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=transforms.ToTensor())

cifar_train, cifar_val = torch.utils.data.random_split(cifar_data,[40000,10000],generator=torch.Generator().manual_seed(420))

train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(cifar_test, batch_size= batch_size, shuffle=False)


network = nn.Sequential(
    
    nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=3),

    nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=3),

    nn.Flatten(),

    nn.Linear(80, 10),
    #nn.Softmax()
)
val_images, val_labels = next(iter(validation_loader))
networkcopy = copy.deepcopy(network)
optimizer = optim.SGD(network.parameters(), lr = 0.01)
loss_function = nn.CrossEntropyLoss()
traininglosses = []
validationlosses = []
firstRun = True

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
        traininglosses.append(loss.item())
        
    prediction = network(val_images)
    new_validationloss = loss_function(prediction, val_labels)

    if firstRun:
        validationloss = new_validationloss
        firstRun = False
    elif  new_validationloss.item() < validationloss.item():
        validationloss = new_validationloss
        networkcopy = copy.deepcopy(network)

    validationlosses.append(validationloss.item())

corr = 0

for index, (image, label) in enumerate(test_loader):
    guess = torch.argmax(network(image), dim=-1)
    corr += (guess == label).sum()
print("\n","Result on test:", corr/10000)
print(traininglosses)