import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mp
import os
import glob
import platform

Image.MAX_IMAGE_PIXELS = None
path = "J:\Final\Scene17_Globe"

joinPathChar = "/"
if(platform.system() == "Windows"):
    joinPathChar = "\\"

os.chdir(path)
my_files1 = glob.glob('*.JPG')


mertens_ar = []

for i in range(1):


    print("i is ", i)

    temp_img_ind = int(i * 15)
    # temp_stack = deepcopy(my_files1[temp_img_ind:temp_img_ind + 15])
    img_ar = []

    for j in range(15):

        print("j is ", j)
        check = os.path.abspath(my_files1[temp_img_ind+j])
        im = cv2.imread(check)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        img_ar.append(im)

    temp_stack = deepcopy(img_ar[0:15])

    # Exposure fusion using Mertens
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(temp_stack)
    print(type(res_mertens))

    # print(type(res_mertens))
    # Convert datatype to 8-bit and save

    res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    cv2.putText(res_mertens_8bit, 'HDR-Mertens', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    mertens_ar.append(res_mertens_8bit)


print(len(mertens_ar))

img = Image.fromarray(mertens_ar[0])

video = cv2.VideoWriter("C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\high_mertens.avi", cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), 1,
                                (img.width, img.height))

for i in range(len(mertens_ar)):

    img = mertens_ar[i]
    print("video product ", i)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    video.write(img)
    cv2.imwrite("C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\HDR_Mertens_Video\\try.jpeg", img)
    #img.save("C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\HDR_Mertens_Video\\try.jpeg")