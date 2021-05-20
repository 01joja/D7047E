
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
#writer2 = SummaryWriter()

valLoss = []
trainLoss = []
with open("networks/epochs_10", 'rb') as f:
    network = pickle.load(f)
    valLoss = network["valLoss"][1:]
    trainLoss = network["trainLoss"]

print(trainLoss)

with open("networks/epochs_20", 'rb') as f:
    network = pickle.load(f)
    valLoss[len(valLoss):] = network["valLoss"][1:]
    trainLoss = network["trainLoss"]

with open("networks/epochs_30", 'rb') as f:
    network = pickle.load(f)
    valLoss[len(valLoss):] = network["valLoss"][1:]
    trainLoss = network["trainLoss"]

with open("networks/epochs_40", 'rb') as f:
    network = pickle.load(f)
    valLoss[len(valLoss):] = network["valLoss"][1:]
    trainLoss = network["trainLoss"]

with open("networks/epochs_50", 'rb') as f:
    network = pickle.load(f)
    valLoss[len(valLoss):] = network["valLoss"][1:]
    trainLoss = network["trainLoss"]

#for i in range(len(trainLoss)):
#    writer.add_scalar('hej/validationLoss', valLoss[i], i)

import matplotlib.pyplot as plt
plt.plot(trainLoss)
plt.plot(valLoss)
plt.ylabel('some numbers')
plt.show()
