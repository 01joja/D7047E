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
    transforms.Resize([360,360]),
    transforms.ToTensor(),
    transforms.Normalize((0.4823), (0.2230)),
])

val2Path = moveDataset.getTestPath()
Dataset =loadDataset.PneumoniaDataSet(val2Path, transform = transform, preprocess=preprocess)
test_loader = DataLoader(Dataset, batch_size= batch_size, shuffle=True)

# Change what network you want to check here.
with open("networks/epochs_80", 'rb') as f:
    object = pickle.load(f)
    temp = {}
    if type(object) == type(temp):
        best_model = object["network"]
    else:
        best_model = object


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
correctness = corr/noImages

print("\n","Result on test:{:2.3%}".format(correctness))
print("Guessed correct sick:", correctSick, "Guessed incorrect sick:", incorrectSick)
print("Guessed correct normal:", correctNormal, "Guessed incorrect normal:", incorrectNormal)
print("Normal correctness: {:2.3%} Sick correctness: {:2.3%} ".format(correctNormal/(correctNormal+incorrectSick), correctSick/(correctSick+incorrectNormal)))
#writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, 'run': 'MNIST Traingin'},
#                    {'hparam/accuracy': correctness})
