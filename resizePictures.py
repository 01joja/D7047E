#To run file
# Linux
# python3 -m pip install --upgrade pip
# python3 -m pip install --upgrade Pillow
# Windows (VS code)
# pip install --upgrade pip
# pip install --upgrade Pillow
import os
from PIL import Image
import getSize
#image = PIL.Image.open("sample.png")

startPath = os.path.join(os.getcwd(),"dataset")
save = os.path.join(os.getcwd(),"modded_dataset")

width, hight = getSize.check_max_size(startPath)

def change_pic(pathLoad,pathSave,width,hight):
    for filename in os.listdir(pathLoad):
        file_path = os.path.join(pathLoad,filename)
        save_file_path = os.path.join(pathSave,filename)
        if os.path.isfile(file_path):
            image = Image.open(file_path)
            w, h = image.size
            result = Image.new(image.mode,(width,hight), 0)
            result.paste(image,(int(width/2-w/2),int(hight/2-h/2)))
            result.save(os.path.join(save_file_path), quality=95)
        else:
            try:
                os.mkdir(save_file_path)
            except:
                print(save_file_path, "allready exist")
            change_pic(file_path,save_file_path,width,hight)

change_pic(startPath,save,width,hight)


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

