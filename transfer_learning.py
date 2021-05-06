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
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

epochs = 1
learning_rate = 0.001

with (open("MNIST_network","rb")) as nw:
    network = pickle.load(nw)

optimizer = optim.Adam(network.parameters(), lr = learning_rate)
best_model = copy.deepcopy(network)
loss_function = nn.CrossEntropyLoss()
validation_loss = 9000

# Stop learning in the original network
for param in network.parameters():
    param.requires_grad=False
# Change the last layer so we get 10 classes.
network[11]=nn.Linear(4096,10)
print(network)

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
    writer.add_scalar('network_fine/traininglosses', new_trainingloss/i, epoch)

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

    writer.add_scalar('network_fine/validationloss', new_validationloss/i, epoch)

# Run on test data
corr = 0
i = 0
for index, (image, label) in enumerate(test_loader):
    i +=1
    guess = torch.argmax(best_model(image), dim=-1)
    result = (guess == label).sum()
    corr += result.item()
    if 1 == (i%30):
        print("\r", "Right guess:", 100*corr/i, "Tested pictures:", 100*i/10000,end="                                                         ")
correctness = 100*corr/10000
print("\n","Result on test:", correctness)
writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                    {'hparam/accuracy': correctness})

# Store the best network
with open("MNIST_network","wb") as handle:
    pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

