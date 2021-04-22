 from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


batch_size = 200
epochs = 1
learning_rate = 0.000001



preprocess = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding = 2),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
    transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
])

preprocessTest = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
    transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
])

MNIST_data = datasets.MNIST("./", train=True, download=True, transform=preprocess)
MNIST_test = datasets.MNIST("./", train=False, download=True, transform=preprocessTest)

MNIST_train, MNIST_val = torch.utils.data.random_split(MNIST_data,[50000,10000],generator=torch.Generator().manual_seed(420))


train_loader = DataLoader(MNIST_train, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(MNIST_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(MNIST_test, batch_size= batch_size, shuffle=False)



def createNetwork():
    return nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,padding=1),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=2),


    nn.Flatten(),
    nn.Linear(4096, 10),
    )

network=createNetwork()

optimizer = optim.Adam(network.parameters(), lr = learning_rate)
best_model = copy.deepcopy(network)
loss_function = nn.CrossEntropyLoss()
validation_loss = 9000

for epoch in range(epochs):
    new_trainingloss = 0
    i = 0
    # Toggle training AKA turing on dropout
    for train_nr, (images, labels) in enumerate(train_loader):
        i += 1
        optimizer.zero_grad()
        prediction = network(images)
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        print(
            '\rEpoch {} [{}/{}] - Loss: {} train'.format(
                epoch+1, train_nr+1, len(train_loader), loss
            ),
            end='                                                 '
        )
        new_trainingloss += loss.item()
    writer.add_scalar('MINST/1e-6', new_trainingloss/i, epoch)

    #Toggle evaluation AKA turing off dropout
    total_val_loss = 0
    i = 0
    for val_nr, (images, labels) in enumerate(validation_loader):
        i += 1
        prediction = network(images)
        loss = loss_function(prediction, labels).item()
        total_val_loss += loss
        print(
            '\rEpoch {} [{}/{}] - Loss: {} val'.format(
                epoch+1, val_nr+1, len(validation_loader), loss
            ),
            end='                                                 '
        )
    
    #Calculate the new_validationloss
    new_validationloss = total_val_loss/i
    if new_validationloss < validation_loss:
        validation_loss = new_validationloss
        best_model = copy.deepcopy(network)

    writer.add_scalar('MINST/1e-6', new_validationloss/i, epoch)

# Run on test data
corr = 0
for index, (image, label) in enumerate(test_loader):
    guess = torch.argmax(best_model(image), dim=-1)
    result = (guess == label).sum()
    corr += result.item()
    print("\r", "Right guess:", 100*corr/i, "Tested pictures:", 100*index/10000,end="                                                         ")
correctness = 100*corr/10000
print("\n","Result on test:", correctness)
writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, 'run': 'MNIST Traingin'},
                    {'hparam/accuracy': correctness})

# Store the best network
with open("1e-6","wb") as handle:
    pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
