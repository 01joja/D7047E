
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
with open("best_network", 'rb') as f:
    network = pickle.load(f)
    valLoss = network["valLoss"]
    trainLoss = network["trainLoss"]


plt.plot(trainLoss,label="hej")
plt.plot(valLoss,label="2")
plt.ylabel('Loss')
plt.xlabel("Epochs")
plt.savefig("networks/loss" + ".pdf",format="pdf")
