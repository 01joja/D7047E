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
save_path_PCA = os.path.join(save_path,'1e-3 PCA.png')
save_path_tSNE = os.path.join(save_path,'1e-3 t-SNE.png')

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

print(batch_size)

print("Load 1e-3 network")
with open('best_network', 'rb') as f:
    network = pickle.load(handle)
network = nn.Sequential(*[network[i] for i in range(11)])

print("Extract features from convolution part")
for train_nr, (images, labels) in enumerate(pictures):
    size = list(network(images).size())
    features = network(images)
    label=labels.detach().numpy()
    break

# Transform the pytorch tensor to numpy and detaches the unwanted information 
np_features=features.detach().numpy()

# Sets 10 colors to the different labels
colors=["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4"]

print("Calculating PCA for 1e-3 network")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(np_features)
print("Procentage of influce from 2 most influential features ", pca.explained_variance_ratio_)

# Used to only add every label once.
# There is probably a better whay of doing it.
print("adding points to plot")
label_added=[False,False,False,False,False,False,False,False,False,False]
fig, ax = plt.subplots()
for i in range(len(label)):
    if label_added[label[i]]:
        ax.scatter(pca_result[i][0], pca_result[i][1], c=colors[label[i]],
               alpha=0.7, edgecolors='none')
    
    else:
        ax.scatter(pca_result[i][0], pca_result[i][1], c=colors[label[i]],  label=label[i],
               alpha=0.7, edgecolors='none')
        label_added[label[i]]=True
    if i%100 == 0:
        print("\r" + str(100*i/batch_size), end="%      ")
print("\nDone!")

ax.legend()
ax.grid(True)
fig.savefig(save_path_PCA, format='png')

print("Processing t-SNE")
tSNE = TSNE(n_components=2)
tSNE_result = tSNE.fit_transform(np_features)


label_added=[False,False,False,False,False,False,False,False,False,False]

print("adding points to plot")
fig, ax = plt.subplots()
for i in range(len(label)):
    if label_added[label[i]]:
        ax.scatter(tSNE_result[i][0], tSNE_result[i][1], c=colors[label[i]],
               alpha=0.7, edgecolors='none')
    
    else:
        ax.scatter(tSNE_result[i][0], tSNE_result[i][1], c=colors[label[i]],  label=label[i],
               alpha=0.7, edgecolors='none')
        label_added[label[i]]=True
    if i%100 == 0:
        print("\r" + str(100*i/batch_size), end="%      ")
            
print("\nDone!")
ax.legend()
ax.grid(True)
fig.savefig(save_path_tSNE, format='png')

print("Saved plots in",save_path)
