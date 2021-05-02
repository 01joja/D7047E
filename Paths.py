import os
from PIL import Image

startPath = os.path.join(os.getcwd(),"dataset")

trainPath = os.path.join(startPath,"train")
trainNORMALPath = os.path.join(trainPath,"NORMAL")
trainPNEUMONIAPath = os.path.join(trainPath,"PNEUMONIA")

testPath = os.path.join(startPath,"test")
testNORMALPath = os.path.join(testPath,"NORMAL")
testPNEUMONIAPath = os.path.join(testPath,"PNEUMONIA")

valPath = os.path.join(startPath,"val")
valNORMALPath = os.path.join(valPath,"NORMAL")
valPNEUMONIAPath = os.path.join(valPath,"PNEUMONIA")


print(valPNEUMONIAPath)
