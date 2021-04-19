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
epochs = 5
learning_rate = 0.0001

# Makes it so that the images fits the network
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loads the data sets and devides them in to train, val and test.
cifar_data = datasets.CIFAR10("./", train=True, download=True, transform=preprocess)
cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=preprocess)
cifar_train, cifar_val = torch.utils.data.random_split(cifar_data,[40000,10000],generator=torch.Generator().manual_seed(420))
train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(cifar_test, batch_size= batch_size, shuffle=False)

# Import alexnet
alex = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)

# Change the last layer so we get 10 classes.
alex.classifier[6]=nn.Linear(4096,10)
print(alex)

# Creates the optimizer
optimizer = optim.Adam(alex.parameters(), lr = learning_rate)
best_model = copy.deepcopy(alex)
loss_function = nn.CrossEntropyLoss()
validation_loss = 9000

for epoch in range(epochs):
    new_trainingloss
    i = 0
    # Toggle training AKA turing on dropout
    alex.train()
    for train_nr, (images, labels) in enumerate(train_loader):
        i += 1
        optimizer.zero_grad()
        prediction = alex(images)
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
    writer.add_scalar('Alex_fine/traininglosses', new_trainingloss/i, epoch)

    #Toggle evaluation AKA turing off dropout
    total_val_loss = 0
    alex.eval()
    i = 0
    for val_nr, (images, labels) in enumerate(validation_loader):
        i += 1
        prediction = alex(images)
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
        best_model = copy.deepcopy(alex)

    writer.add_scalar('Alex_fine/validationloss', new_validationloss/i, epoch)

# Run on test data
corr = 0
i = 0
for index, (image, label) in enumerate(test_loader):
    guess = torch.argmax(best_model(image), dim=-1)
    result = (guess == label).sum()
    corr += result.item()
    if 0 == (i%30):
        print("\r", "Right guess:", 100*corr/i, "Tested pictures:", 100*i/10000,end="                                                         ")
correctness = 100*corr/10000
print("\n","Result on test:", correctness)
writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                    {'hparam/accuracy': correctness})

# Store the best network
save_file = open("fine_tune_Alex_network","wb")
pickle.dump(save_file,best_model,correctness)
save_file.close()

                    