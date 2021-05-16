import os
from PIL import Image, ImageOps
from skimage import io, transform
import os, shutil
import numpy as np

# Moves all pictures to correct folder to make sure Train, Val1, Val2 and Test
# allways are the same. 
# 
# Val2 is as a testset for during development. Test should only be touched once.
# 
# The original data should be in the folders "dataset/test" and 
# "dataset/train". Train and test should have the subfolders "NORMAL" and 
# "PNEUMONIA"

# Returns the path for the folders Train. Val1, Val2, and Test
# val1N = number of training pictures that should be put in Validation 1 Normal
# val1P = same but for PNEUMONIA
# val2N = Validation 2 NORMAL
# val2P = Validation 2 PNEUMONIA
def moveDataset(val1N = 154,val1P = 462, val2N = 154, val2P = 462):
    #Sets up all paths and creates folders for them
    load = os.path.join(os.getcwd(),"dataset")
    save = os.path.join(os.getcwd(),"modded_dataset")
    try:
        # tries to remove everything in modded_dataset folder
        shutil.rmtree(save, ignore_errors = False)
    except:
        print("")
    os.mkdir(save)
    
    saveTrain = os.path.join(save,"train")
    os.mkdir(saveTrain)
    saveTrainN = os.path.join(saveTrain,"NORMAL")
    os.mkdir(saveTrainN)
    saveTrainP = os.path.join(saveTrain,"PNEUMONIA")
    os.mkdir(saveTrainP)

    saveVal1  = os.path.join(save,"val1")
    os.mkdir(saveVal1)
    saveVal1N = os.path.join(saveVal1,"NORMAL")
    os.mkdir(saveVal1N)
    saveVal1P = os.path.join(saveVal1,"PNEUMONIA")
    os.mkdir(saveVal1P)

    saveVal2  = os.path.join(save,"val2")
    os.mkdir(saveVal2)
    saveVal2N = os.path.join(saveVal2,"NORMAL")
    os.mkdir(saveVal2N)
    saveVal2P = os.path.join(saveVal2,"PNEUMONIA")
    os.mkdir(saveVal2P)

    saveTest  = os.path.join(save,"test")
    os.mkdir(saveTest)
    saveTestN = os.path.join(saveTest,"NORMAL")
    os.mkdir(saveTestN)
    saveTestP = os.path.join(saveTest,"PNEUMONIA")
    os.mkdir(saveTestP)

    loadTrain = os.path.join(load,"train")
    loadTrainN = os.path.join(loadTrain,"NORMAL")
    loadTrainNList = os.listdir(loadTrainN)
    loadTrainP = os.path.join(loadTrain,"PNEUMONIA")
    loadTrainPList = os.listdir(os.path.join(loadTrain,"PNEUMONIA"))

    loadTest  = os.path.join(load,"test")
    loadTestN = os.path.join(loadTest,"NORMAL")
    loadTestNList = os.listdir(loadTestN)
    loadTestP = os.path.join(loadTest,"PNEUMONIA")
    loadTestPList = os.listdir(os.path.join(loadTest,"PNEUMONIA"))

    # Iterates through everyting in dataset/tran/NORMAL
    # saves them in: 
    # modded_dataset/val1/NORMAL
    # modded_dataset/val2/NORMAL
    # modded_dataset/train/NORMAL
    i=0
    for filename in loadTrainNList:
        if filename != ".DS_Store":
            file_path = os.path.join(loadTrainN,filename)
            filenameV2, _ = filename.split(".")
            filenameV2 = filenameV2+"-Augmented.jpeg"
            #Chooses the dir to save 
            augment = True
            if i < val1N:
                save_file_path = os.path.join(saveVal1N,filename)
                save_file_pathV2 = os.path.join(saveVal1N,filenameV2)
                augment = False
            elif i < val1N+val2N:
                save_file_path = os.path.join(saveVal2N,filename)
                save_file_pathV2 = os.path.join(saveVal2N,filenameV2)
                augment = False
            else:
                save_file_path = os.path.join(saveTrainN,filename)   
                save_file_pathV2 = os.path.join(saveTrainN,filenameV2)        
            shutil.copyfile(file_path, save_file_path)

            if augment:
                # Horizontal flip of the image
                image = Image.open(file_path)
                result = ImageOps.mirror(image)
                result.save(save_file_pathV2, quality=95)
            i += 1
    
    # Iterates through everyting in dataset/tran/PNEUMONIA
    # saves them in: 
    # modded_dataset/val1/PNEUMONIA
    # modded_dataset/val2/PNEUMONIA
    # modded_dataset/train/PNEUMONIA
    i=0
    for filename in loadTrainPList:
        file_path = os.path.join(loadTrainP,filename)
        if filename != ".DS_Store":
            if i < val1P:
                save_file_path = os.path.join(saveVal1P,filename)
            elif i < val1P+val2P:
                save_file_path = os.path.join(saveVal2P,filename)
            else:
                save_file_path = os.path.join(saveTrainP,filename)            
            shutil.copyfile(file_path, save_file_path)
            i += 1
    
    # Moves the testdata for the NORMAL.
    for filename in loadTestNList: 
        file_path = os.path.join(loadTestN,filename)
        if filename != ".DS_Store":
            save_file_path = os.path.join(saveTestN,filename)            
            shutil.copyfile(file_path, save_file_path)
    
    # Moves the testdata for the PNEUMONIA.
    for filename in loadTestPList:
        file_path = os.path.join(loadTestP,filename)
        if filename != ".DS_Store":
            save_file_path = os.path.join(saveTestP,filename)            
            shutil.copyfile(file_path, save_file_path)

def getTrainPath():
    path = os.path.join(os.getcwd(),"modded_dataset")
    path = os.path.join(path,"train")
    return path

def getTestPath():
    path = os.path.join(os.getcwd(),"modded_dataset")
    path = os.path.join(path,"test")
    return path

def getVal1Path():
    path = os.path.join(os.getcwd(),"modded_dataset")
    path = os.path.join(path,"val1")
    return path

def getVal2Path():
    path = os.path.join(os.getcwd(),"modded_dataset")
    path = os.path.join(path,"val2")
    return path

