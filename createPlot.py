
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

valLoss = []
trainLoss = []
with open("networks/epochs_10", 'rb') as f:
    network = pickle.load(f)
    valLoss = network["valLoss"][1:]
    trainLoss = network["trainLoss"]


plt.plot(trainLoss)
plt.plot(valLoss)
plt.ylabel('Loss')
plt.xlabel("Epochs")
plt.savefig("networks/loss" + ".pdf",format="pdf")
