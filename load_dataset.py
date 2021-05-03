# This is the example of how to do it on pytorch https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

# Look a this https://discuss.pytorch.org/t/how-to-load-images-without-using-imagefolder/59999/2
# The original code uses a CSV
# needed these packages:
#   scikit-image
#   pandas


# Needed import to get the example to work
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, io
from PIL import Image


'''
# Just to show one image
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('faces/', img_name)),
               landmarks)
# This stops the code plt.show()
# This only shows the plot and continues the code plt.draw()
# Use plt.show() at the end of code to ensure that the windows stay opened.
plt.draw()
'''


# If no transform given it will just return the picture as tensor
class PneumoniaDataSet(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        pneumonia_dir = os.path.join(main_dir,"PNEUMONIA")
        normal_dir = os.path.join(main_dir,"NORMAL")
        pneumonia_imgs = os.listdir(pneumonia_dir)
        normal_imgs = os.listdir(normal_dir)
        self.total_imgs = len(pneumonia_imgs) + len(normal_imgs)
        self.path_to_pictures=[""]*self.total_imgs
        self.labels = [torch.tensor([0,1])]*self.total_imgs
        for i in range(self.total_imgs):
            if i < len(normal_imgs):
                self.path_to_pictures[i]=os.path.join(normal_dir,normal_imgs[i])
                normal = torch.tensor([1,0])
                self.labels[i] = normal
            else: 
                self.path_to_pictures[i]=os.path.join(normal_dir,pneumonia_imgs[i-len(normal_imgs)])

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, idx):
        img_loc = self.path_to_pictures[idx]
        if self.transform:
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
        else:
            tensor_image = io.read_image(img_loc)
        return tensor_image


startPath = os.path.join(os.getcwd(),"dataset")
startPath = os.path.join(startPath,"train")

train = PneumoniaDataSet(startPath)
for i in range(train.__len__()):
    train.__getitem__(i)


