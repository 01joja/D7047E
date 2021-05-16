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
import moveDataset
import os
from datetime import datetime, timedelta

batch_size = 1
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

val2Path = moveDataset.getVal2Path()
Dataset =loadDataset.PneumoniaDataSet(val2Path, transform = transform, preprocess=preprocess)
test_loader = DataLoader(Dataset, batch_size= batch_size, shuffle=True)

with open("networks/network_epochs_40_batch_200", 'rb') as f:
            best_model = pickle.load(f)

# Run on test data
corr = 0
guesses = 0
correctSick = 0
incorrectSick = 0
correctNormal = 0
incorrectNormal = 0
noImages = Dataset.__len__()
starT = datetime.now()

for index, (image, label,_) in enumerate(test_loader):
    
    guess = torch.argmax(best_model(image), dim=-1)
    result = (guess == label).sum()
    corr += result.item()
    guesses +=image.size()[0]
    nowT = datetime.now()
    deltaT =  nowT - starT
    tLeft = deltaT*(1/(guesses/noImages)-1)

    if guess.item()==1:
        if label.item() == 1:
            correctSick+=1
        else:
            incorrectSick+=1
    else:
        if label.item() == 0:
            correctNormal+=1
        else: 
            incorrectNormal+=1
    print("\r", "Right guess: {:3.2%}".format(corr/guesses), "Tested pictures: {:3.2%}".format(guesses/noImages) ,
        end="                 Time left: {} ".format(tLeft)
    )
correctness = 100*corr/noImages
print("\n","Result on test:", correctness)
print("Guessed correct sick:", correctSick, "Guessed incorrect sick:", incorrectSick)
print("Guessed correct normal:", correctNormal, "Guessed incorrect normal:", incorrectNormal)
#writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, 'run': 'MNIST Traingin'},
#                    {'hparam/accuracy': correctness})
