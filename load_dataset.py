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
class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        self.total_imgs = len(self.all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        if self.transform:
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
        else:
            tensor_image = io.read_image(img_loc)
        return tensor_image


startPath = os.path.join(os.getcwd(),"dataset")
startPath = os.path.join(startPath,"train")
startPath = os.path.join(startPath,"NORMAL")

test = CustomDataSet(startPath)
print(test.total_imgs)
print(test.__getitem__(1))

