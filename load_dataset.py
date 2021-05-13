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
import pickle



# If no transform given it will just return the picture as tensor
class PneumoniaDataSet(Dataset):
    # mainDir is the path to the main directory where the test/train files are.
    # Preprocess: if True it will do the transformation before __getitem__
    def __init__(self, mainDir, transform=None, targetTransform=None, preprocess = False):
        self.no = 0
        self.preprocess = preprocess
        # This dose not work.
        #Will transform the pictures and store them in pickle format in a map next to the mainDir
        def transfromPic(mainDir,transform):
            originalDir = mainDir
            
            # Cereate a new mainDir + "_transformed" as a dir.
            path_to_mainDir, mainDir = os.path.split(mainDir)
            saveDir = mainDir+"_transformed"
            mainDir = os.path.join(path_to_mainDir,saveDir)

            
            # Cleans up the file path and creates the path where the 
            # transforms will be stored.
            try:
                shutil.rmtree(mainDir, ignore_errors = False)
            except:
                print("No file to remove")
                print(mainDir)
            finally:
                os.mkdir(mainDir)
                sick = os.path.join(mainDir,"PNEUMONIA")
                os.mkdir(sick)
                originalSick = os.path.join(originalDir,"PNEUMONIA")
                normal = os.path.join(mainDir,"NORMAL")
                os.mkdir(normal)
                originalNormal = os.path.join(originalDir,"NORMAL")
            
            #Help function for the for-loops.
            def copyTransformAndPickle(loadDic,saveDic,imageName):
                # Change image file end from .jpeg to .pt
                newName = imageName.split(".")[0] + ".pt"
                loadPath = os.path.join(loadDic,imageName)
                savePath = os.path.join(saveDic,newName)
                image = Image.open(loadPath).convert("RGB")
                tensor_image = transform(image)
                torch.save(tensor_image, savePath)
                #with open(savePath, 'wb') as f:
                #    pickle.dump(tensor_image, f, protocol=pickle.HIGHEST_PROTOCOL)

            length = len(os.listdir(originalSick)) + len(os.listdir(originalNormal))
            transfromed = 0
            print("Transforming pictures")
            for fileName in os.listdir(originalSick):
                copyTransformAndPickle(originalSick,sick,fileName)
                transfromed +=1
                procentage = "{:.4f}".format(transfromed/length)
                print(
                    '\rTransforming {} %'.format(
                        procentage
                    ),
                    end='                                                 '
                )
            for fileName in os.listdir(originalNormal):
                copyTransformAndPickle(originalNormal,normal,fileName)   
                transfromed +=1
                procentage = "{:.4f}".format(transfromed/length)
                print(
                    '\rTransforming {} %'.format(
                        procentage
                    ),
                    end='                                                 '
                )
            return mainDir     
        if transform and preprocess:
            mainDir=transfromPic(mainDir,transform)
        self.transform = transform
        self.targetTransform = targetTransform
        pneumoniaDir = os.path.join(mainDir,"PNEUMONIA")
        normalDir = os.path.join(mainDir,"NORMAL")
        pneumoniaImgs = os.listdir(pneumoniaDir)
        normalImgs = os.listdir(normalDir)
        self.totalImgs = len(pneumoniaImgs) + len(normalImgs)
        self.path_to_pictures=[""]*self.totalImgs
        # Creates a list with the label 1 (PNEUMONIA) of size = self.totalImags 
        self.labels = [1]*self.totalImgs
        for i in range(self.totalImgs):
            if i < len(normalImgs):
                self.path_to_pictures[i]=os.path.join(normalDir,normalImgs[i])
                # Makes the label correct for all normal images.
                self.labels[i] = 0
            else: 
                # i - len(normalImags) is there to get the first index of the second list
                self.path_to_pictures[i]=os.path.join(pneumoniaDir,pneumoniaImgs[i-len(normalImgs)])

        

    def __len__(self):
        return self.totalImgs

    def __getitem__(self, idx):
        img_loc = self.path_to_pictures[idx]
        _, imageName = os.path.split(img_loc)
        self.no+=1
        #print(img_loc)
        label = self.labels[idx]
        if self.transform:
            if self.preprocess:
                tensor_image = torch.load(img_loc)
                #with open(img_loc, 'rb') as f:
                #    tensor_image = pickle.load(f)
            else:
                image = Image.open(img_loc).convert("RGB")
                tensor_image = self.transform(image)
        else:
            tensor_image = io.read_image(img_loc)
        if self.targetTransform:
            label = self.targetTransform(label)
        sample = [tensor_image, label, imageName]
        return sample

    def getTimesRun(self):
        return self.no
    

def getTrainPath():
    path = os.path.join(os.getcwd(),"dataset")
    path = os.path.join(path,"train")
    return path

def getTestPath():
    path = os.path.join(os.getcwd(),"dataset")
    path = os.path.join(path,"test")
    return path

def testPath():
    path = os.path.join(os.getcwd(),"modded_dataset")
    path = os.path.join(path,"val")
    return path


preprocessTraining = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(2500),
    transforms.RandomCrop(2500, padding=100),
    transforms.ToTensor(),
    transforms.Normalize((0.4823), (0.2230)),
])

#mainDir = testPath()
#test = PneumoniaDataSet(mainDir, transform = preprocessTraining)
#print(test.__getitem__(0))

