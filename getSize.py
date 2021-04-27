#To run file
# Linux
# python3 -m pip install --upgrade pip
# python3 -m pip install --upgrade Pillow
# Windows (VS code)
# pip install --upgrade pip
# pip install --upgrade Pillow
import os
from PIL import Image
#image = PIL.Image.open("sample.png")

startPath = os.path.join(os.getcwd(),"dataset")


def check_max_size(path):
    l_width = 0
    l_height = 0
    width = -1
    height = -1
    for filename in os.listdir(path):
        file_path = os.path.join(path,filename)
        if os.path.isfile(file_path):
            if filename != ".DS_Store":
                image = Image.open(file_path)
                width, height = image.size
        else:
            width, height = check_max_size(file_path)
        if width>l_width:
            l_width = width
        if height>l_height:
            l_height = height
    return l_width,l_height


w,h =check_max_size(startPath)
print(w)
print(h)

f = open("biggestPicture.txt", "w")
f.write("Largest width: "+str(w)+" Largest height: "+ str(h) + " According to PILLOW")
f.close()


