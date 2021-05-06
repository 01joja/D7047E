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
import os, shutil



# If no transform given it will just return the picture as tensor
class PneumoniaDataSet(Dataset):
    def __init__(self, main_dir, transform=None, target_transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.target_transform = target_transform
        pneumonia_dir = os.path.join(main_dir,"PNEUMONIA")
        normal_dir = os.path.join(main_dir,"NORMAL")
        pneumonia_imgs = os.listdir(pneumonia_dir)
        normal_imgs = os.listdir(normal_dir)
        self.total_imgs = len(pneumonia_imgs) + len(normal_imgs)
        self.path_to_pictures=[""]*self.total_imgs
        self.labels = [1]*self.total_imgs
        for i in range(self.total_imgs):
            if i < len(normal_imgs):
                self.path_to_pictures[i]=os.path.join(normal_dir,normal_imgs[i])
                self.labels[i] = 0
            else: 
                self.path_to_pictures[i]=os.path.join(pneumonia_dir,pneumonia_imgs[i-len(normal_imgs)])

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, idx):
        img_loc = self.path_to_pictures[idx]
        label = self.labels[idx]
        if self.transform:
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
        else:
            tensor_image = io.read_image(img_loc)
        if self.target_transform:
            label = self.target_transform(label)
        sample = [tensor_image, label]
        return sample


def getTrainPath():
    path = os.path.join(os.getcwd(),"dataset")
    path = os.path.join(path,"train")
    return path

def getTestPath():
    path = os.path.join(os.getcwd(),"dataset")
    path = os.path.join(path,"test")
    return path

#train = PneumoniaDataSet(startPath)


