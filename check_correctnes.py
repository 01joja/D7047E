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

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_testtest, batch_size= batch_size, shuffle=True)

with open("best_network", 'rb') as f:
            best_model = pickle.load(f)

# Run on test data
corr = 0
guesses = 0
correctSick = 0
incorrectSick = 0
correctNormal = 0
incorrectNormal = 0

for index, (image, label,_) in enumerate(test_loader):
    guess = torch.argmax(best_model(image), dim=-1)
    result = (guess == label).sum()
    corr += result.item()
    guesses +=image.size()[0]
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
    print("\r", "Right guess: {:3.2%}".format(corr/guesses), "Tested pictures: {:3.2%}".format(guesses/616) ,end="                                                         ")
correctness = 100*corr/616
print("\n","Result on test:", correctness)
print("Guessed correct sick:", correctSick, "Guessed incorrect sick:", incorrectSick)
print("Guessed correct normal:", correctNormal, "Guessed incorrect normal:", incorrectNormal)
#writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, 'run': 'MNIST Traingin'},
#                    {'hparam/accuracy': correctness})
