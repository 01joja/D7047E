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
import load_dataset
import os

batch_size = 20
epochs = 2
learning_rate = 0.001

preprocessTraining = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(2500),
    transforms.RandomCrop(2500, padding=100),
    transforms.ToTensor(),
    transforms.Normalize((0.4823), (0.2230)),
])

print("loading dataset")
path = load_dataset.getTrainPath()
Dataset =load_dataset.PneumoniaDataSet(path, transform = preprocessTraining)
data_train, data_valtest = torch.utils.data.random_split(Dataset,[4000,1232],generator=torch.Generator().manual_seed(420))
data_val, data_testtest = torch.utils.data.random_split(data_valtest,[616,616],generator=torch.Generator().manual_seed(420))


train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_testtest, batch_size= batch_size, shuffle=True)

def createNetwork():
    return nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=128, out_channels=126, kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),



    nn.Flatten(),
    nn.Linear(504, 4096),
    nn.Linear(4096,2)
    )

print("creates dataset")
network = createNetwork()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network.to(device)
optimizer = optim.Adam(network.parameters(), lr = learning_rate)
best_model = copy.deepcopy(network)
loss_function = nn.CrossEntropyLoss()
validation_loss = 9000

print("start traning")
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
    writer.add_scalar('MINST/traininglosses', new_trainingloss/i, epoch)

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

    writer.add_scalar('MINST/validationloss', new_validationloss/i, epoch)