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
import loadDataset
import os
from datetime import datetime, timedelta
import moveDataset


batch_size = 20
epochs = 10
learning_rate = 0.001

#gets the paths to the different datasets
val1Path = moveDataset.getVal1Path()
val2Path = moveDataset.getVal2Path()
trainPath = moveDataset.getTrainPath()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize([400,400]),
    transforms.RandomCrop([350,350], padding=30),
    transforms.ToTensor(),
    transforms.Normalize((0.4823), (0.2230)),
])

continueTraning = False
testDrive = False
while True:
    val = input("Continue training?(Yes/No):")
    if val == "Yes":
        continueTraning = True
        break
    elif val == "No":
        continueTraning = False
        break
    print("Answer Yes or No")
print("Load dataset")
try:
    val1 = loadDataset.PneumoniaDataSet(val1Path, transform = transform)
    val2 = loadDataset.PneumoniaDataSet(val2Path, transform = transform)
    train = loadDataset.PneumoniaDataSet(trainPath, transform = transform)
except:
    print("Creates new dataset")
    moveDataset.moveDataset(val1N = 154,val1P = 462, val2N = 308, val2P = 462)
    val1 = loadDataset.PneumoniaDataSet(val1Path, transform = transform)
    val2 = loadDataset.PneumoniaDataSet(val2Path, transform = transform)
    train = loadDataset.PneumoniaDataSet(trainPath, transform = transform)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(val1, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val2, batch_size= batch_size, shuffle=True)

def createNetwork():
    return nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5,padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    #nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
    #nn.ReLU(),
    #nn.Conv2d(in_channels=128, out_channels=126, kernel_size=3,padding=1),
    #nn.ReLU(),
    #nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(29584, 4096),
    nn.Linear(4096,2)
    )
try:
    if continueTraning:
        print("Loading network")
        with open("best_network", 'rb') as f:
            network = pickle.load(f)
except:
    continueTraning = False
    print("You don't have an existing network")

if continueTraning == False:
    print("Creates new network")
    network = createNetwork()
optimizer = optim.Adam(network.parameters(), lr = learning_rate)
best_model = copy.deepcopy(network)
loss_function = nn.CrossEntropyLoss()
validation_loss = 9000


totalElements = epochs*val1.__len__()+epochs*train.__len__()
elementsDone = 0
starT = datetime.now()

print("Training started:",starT,"\n")
for epoch in range(epochs):
    new_trainingloss = 0
    # Toggle training AKA turing on dropout
    for train_nr, (images, labels,_) in enumerate(train_loader):
        elementsDone +=images.size()[0]
        optimizer.zero_grad()
        prediction = network(images)
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        nowT = datetime.now()
        deltaT =  nowT - starT
        tLeft = deltaT*(1/(elementsDone/totalElements)-1)
        print(
            '\rEpoch {:3}/{:3} [{:5}/{:5}] - Loss: {:3.4} train'.format(
                epoch+1,epochs, train_nr+1, len(train_loader), loss
            ),
            end='                         Done: {:2.3%} Time left: {} '.format(elementsDone/totalElements, tLeft)
        )
        new_trainingloss += loss.item()
    #swriter.add_scalar('MINST/traininglosses', new_trainingloss/i, epoch)

    #Toggle evaluation AKA turing off dropout
    i = 0
    total_val_loss = 0
    for val_nr, (images, labels,_) in enumerate(validation_loader):
        i+=1
        elementsDone +=images.size()[0]
        prediction = network(images)
        loss = loss_function(prediction, labels).item()
        total_val_loss += loss
        nowT = datetime.now()
        deltaT =  nowT - starT
        tLeft = deltaT*(1/(elementsDone/totalElements)-1)
        print(
            '\rEpoch {:3}/{:3} [{:5}/{:5}] - Loss: {:3.4} train'.format(
                epoch+1,epochs, train_nr+1, len(train_loader), loss
            ),
            end='                         Done: {:2.3%} Time left: {} '.format(elementsDone/totalElements, tLeft)
        )
    
    
    #Calculate the new_validationloss
    new_validationloss = total_val_loss/i
    if new_validationloss < validation_loss:
        validation_loss = new_validationloss
        best_model = copy.deepcopy(network)
        # Saves network if the loss where better on the validation
        # compered to last validaton.
        print("\r New best! loss:" + str(validation_loss), end="                  ")
        with open("best_network", 'wb') as f:
            pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    #writer.add_scalar('MINST/validationloss', new_validationloss/i, epoch)

