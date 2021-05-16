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

batchSize = 200
epochs = 10
learningRate = 0.001

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

transformVal = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize([350,350]),
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
    val1 = loadDataset.PneumoniaDataSet(val1Path, transform = transformVal)
    val2 = loadDataset.PneumoniaDataSet(val2Path, transform = transformVal)
    train = loadDataset.PneumoniaDataSet(trainPath, transform = transform)
except:
    print("Creates new dataset")
    moveDataset.moveDataset(val1N = 154,val1P = 462, val2N = 308, val2P = 462)
    val1 = loadDataset.PneumoniaDataSet(val1Path, transform = transformVal)
    val2 = loadDataset.PneumoniaDataSet(val2Path, transform = transformVal)
    train = loadDataset.PneumoniaDataSet(trainPath, transform = transform)

trainLoader = DataLoader(train, batch_size=batchSize, shuffle=True)
validationLoader = DataLoader(val1, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(val2, batch_size= batchSize, shuffle=True)

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
            object = pickle.load(f)
            temp = {}
            if type(object) == type(temp):
                network = object["network"]
            else:
                network = object
except:
    continueTraning = False
    print("You don't have an existing network")

if continueTraning == False:
    print("Creates new network")
    network = createNetwork()
optimizer = optim.Adam(network.parameters(), lr = learningRate)
bestModel = copy.deepcopy(network)
lossFunction = nn.CrossEntropyLoss()
validationLoss = 9000


totalElements = epochs*val1.__len__()+epochs*train.__len__()
elementsDone = 0
starT = datetime.now()
trainingLoss = [0]
valLoss = [0]

print("Training started:",starT,"\n")
for epoch in range(epochs):
    newTrainingloss = 0
    # Toggle training AKA turing on dropout
    i = 0
    for train_nr, (images, labels,_) in enumerate(trainLoader):
        i+=1
        elementsDone +=images.size()[0]
        optimizer.zero_grad()
        prediction = network(images)
        loss = lossFunction(prediction, labels)
        loss.backward()
        optimizer.step()
        nowT = datetime.now()
        deltaT =  nowT - starT
        tLeft = deltaT*(1/(elementsDone/totalElements)-1)
        print(
            '\rEpoch {:3}/{:3} [{:5}/{:5}] - Loss: {:3.4} train'.format(
                epoch+1,epochs, train_nr+1, len(trainLoader), loss
            ),
            end='                         Done: {:2.3%} Time left: {} '.format(elementsDone/totalElements, tLeft)
        )
        newTrainingloss += loss.item()
    trainingLoss.append(newTrainingloss/i)
    #swriter.add_scalar('MINST/traininglosses', newTrainingloss/i, epoch)

    #Toggle evaluation AKA turing off dropout
    i = 0
    totalValLoss = 0
    for val_nr, (images, labels,_) in enumerate(validationLoader):
        i+=1
        elementsDone +=images.size()[0]
        prediction = network(images)
        loss = lossFunction(prediction, labels).item()
        totalValLoss += loss
        nowT = datetime.now()
        deltaT =  nowT - starT
        tLeft = deltaT*(1/(elementsDone/totalElements)-1)
        print(
            '\rEpoch {:3}/{:3} [{:5}/{:5}] - Loss: {:3.4} train'.format(
                epoch+1,epochs, train_nr+1, len(trainLoader), loss
            ),
            end='                         Done: {:2.3%} Time left: {} '.format(elementsDone/totalElements, tLeft)
        )
    
    #Calculate the newValidationloss
    newValidationloss = totalValLoss/i
    valLoss.append(newValidationloss)

    if newValidationloss < validationLoss:
        validationLoss = newValidationloss
        bestModel = copy.deepcopy(network)
        saveObject = {
            "network": bestModel,
            "valLoss": valLoss,
            "testLoss": trainingLoss
        }
        # Saves network if the loss where better on the validation
        # compered to last validaton.
        print("\r New best! loss:{:3.4}".format(validationLoss), end="                  ")
        with open("best_network", 'wb') as f:
            pickle.dump(saveObject, f, protocol=pickle.HIGHEST_PROTOCOL)

    #writer.add_scalar('MINST/validationloss', newValidationloss/i, epoch)

