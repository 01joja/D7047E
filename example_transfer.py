from torch.utils.tensorboard import SummaryWriter

import os
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

with open('example_network', 'rb') as handle:
    network = pickle.load(handle)

batch_size = 200
learning_rate = 0.001
epochs = 1

preprocessTest = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
])

SVHN_test = datasets.SVHN("./", split='test', download=True, transform=preprocessTest)
SVHN_train = datasets.SVHN("./", split='train', download=True, transform=preprocessTest)

SVHN_train, SVHN_val = torch.utils.data.random_split(SVHN_train,[60000,13257],generator=torch.Generator().manual_seed(420))

test_loader = DataLoader(SVHN_test, shuffle=False)
validation_loader = DataLoader(SVHN_val, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(SVHN_train, batch_size, shuffle=False)

# Stop learning in the original network
for param in network.parameters():
    param.requires_grad=False

network[11]=nn.Linear(4096,10)

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
    writer.add_scalar('MNIST_feature_extract/traininglosses', new_trainingloss/i, epoch)

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

    writer.add_scalar('MNIST_feature_extract/validationloss', new_validationloss/i, epoch)

# Run on test data
corr = 0

for index, (image, label) in enumerate(test_loader):
    guess = torch.argmax(best_model(image), dim=-1)
    result = (guess == label).sum()
    corr += result.item()
    print("\r", "Right guess:", 100*corr/26032, "Tested pictures:", 100*index/26032,end="                                                         ")
correctness = 100*corr/26032
print("\n","Result on test:", correctness)
writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, 'run': 'Feature extracted SVHN'},
                    {'hparam/accuracy': correctness})
                    
# Store the best network
with open("networks/Feature_extract_MNIST_on_SVHN_network","wb") as handle:
    pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

