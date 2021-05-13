# Dependency 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Follow link to install 
# https://scikit-learn.org/stable/install.html

# Command for windows not cuda
# pip install -U scikit-learn

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import pickle
import os


import load_dataset

# Set up savepaths
cwd = os.getcwd()
save_path = os.path.join(cwd,'Results')
save_path_tSNE = os.path.join(save_path,'featureExtract.png')

#Transform the images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize([400,400]),
    transforms.RandomCrop([350,350], padding=30),
    transforms.ToTensor(),
    transforms.Normalize((0.4823), (0.2230)),
])

print("Load dataset")
path = load_dataset.getTrainPath()
Dataset =load_dataset.PneumoniaDataSet(path, transform = transform)

batch_size = Dataset.__len__()
train_loader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)



print(batch_size)

print("Load 1e-3 network")
with open('best_network', 'rb') as f:
    network = pickle.load(f)

network = nn.Sequential(*[network[i] for i in range(11)])
print(network)

print("Extract features from convolution part")
for train_nr, (images, _, names) in enumerate(train_loader):
    
    labels = [""]*len(names)
    i = 0
    for name in names:
        if "virus" in name:
            labels[i]="virus"
        elif "bacteria" in name:
            labels[i]="bacteria"
        else:
            labels[i]="normal"
        i+=1
    features = network(images)
    break

# Transform the pytorch tensor to numpy and detaches the unwanted information 
np_features=features.detach().numpy()

# Sets 3 colors to the different labels
colors=["#e6194B","#3cb44b","#ffe119"]

print("Lowering fetuses to 4000 with PCA")
#pca = PCA(n_components=4000)
#pca_result = pca.fit_transform(np_features)
#print("Procentage of influce from 2 most influential features ", pca.explained_variance_ratio_)

print("Now doing the same thing with t-SNE")
tSNE = TSNE(n_components=2)
tSNE_result = tSNE.fit_transform(np_features)


label_added=[False,False,False]

print("adding points to plot")
fig, ax = plt.subplots()
for i in range(len(labels)):
    label = 0
    if labels[i] == "virus":
        label = 0
    elif labels == "bacteria":
        label = 1
    else:
        label = 2
    if label_added[label]:
        ax.scatter(tSNE_result[i][0], tSNE_result[i][1], c=colors[label],
               alpha=0.7, edgecolors='none')
    
    else:
        ax.scatter(tSNE_result[i][0], tSNE_result[i][1], c=colors[label],  label=labels[i],
               alpha=0.7, edgecolors='none')
        label_added[label]=True
    if i%100 == 0:
        print("\r" + str(100*i/batch_size), end="%      ")
            
print("\nDone!")
ax.legend()
ax.grid(True)
fig.savefig(save_path_tSNE, format='png')

print("Saved plots in",save_path)
