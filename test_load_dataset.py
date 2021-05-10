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

batch_size = 200
epochs = 10
learning_rate = 0.001

preprocess = False
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
    elif val == "test": #Makes the code run the small validation test.
        testDrive == True
        break
    print("Answer Yes or No")


print("loading dataset")
if testDrive==False:
    if preprocess:
        path = load_dataset.getTrainPath()
        Dataset =load_dataset.PneumoniaDataSet(path, transform = transform, preprocess=preprocess)
    else:
        try:
            with open("transformed_dataset", 'rb') as f:
                Dataset = pickle.load(f)
        except:
            path = load_dataset.getTrainPath()
            Dataset =load_dataset.PneumoniaDataSet(path, transform = transform, preprocess=preprocess)
            with open("transformed_dataset", 'wb') as f:
                pickle.dump(Dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    data_train, data_valtest = torch.utils.data.random_split(Dataset,[4000,1232],generator=torch.Generator().manual_seed(420))
    data_val, data_testtest = torch.utils.data.random_split(data_valtest,[616,616],generator=torch.Generator().manual_seed(420))
else:
    path = load_dataset.testPath()
    Dataset =load_dataset.PneumoniaDataSet(path, transform = transform,preprocess=preprocess)
    data_train, data_valtest = torch.utils.data.random_split(Dataset,[8,8],generator=torch.Generator().manual_seed(420))
    data_val, data_testtest = torch.utils.data.random_split(data_valtest,[4,4],generator=torch.Generator().manual_seed(420))



train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_testtest, batch_size= batch_size, shuffle=True)

def createNetwork():
    return nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5,padding=1),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5,padding=1),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3,padding=1),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,padding=1),
    nn.LeakyReLU(),
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
        print("\nLoading network")
        with open("best_network", 'rb') as f:
            network = pickle.load(f)
except:
    continueTraning = False
    print("You don't have an existing network")

if continueTraning == False:
    print("\nCreates new network")
    network = createNetwork()
print("network loaded")
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
    #swriter.add_scalar('MINST/traininglosses', new_trainingloss/i, epoch)

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
        # Saves network if the loss where better on the validation
        # compered to last validaton.
        print("\r New best! loss:" + str(validation_loss), end="                  ")
        with open("best_network", 'wb') as f:
            pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    #writer.add_scalar('MINST/validationloss', new_validationloss/i, epoch)

print(Dataset.getTimesRun())