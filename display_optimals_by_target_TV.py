import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from copy import deepcopy
import cv2
# import face_recognition
import matplotlib.pyplot as plt
import matplotlib as mp

mp.rcParams.update({'axes.titlesize': 14, 'font.size': 11, 'font.family': 'DejaVu Sans'})

import random
import threading

topx, topy, botx, boty = 0, 0, 0, 0
rect_id = None
ii = 0


def get_mouse_posn(event):
    global centy, centx
    global rect_id
    global topy, topx, botx, boty
    centx, centy = event.x, event.y
    topy = max(centy - 4, 0)
    topx = max(centx - 4, 0)
    botx = min(centx + 4, 230)
    boty = min(centy + 4, 300)
    canvas.coords(rect_id, topx, topy, botx, boty)  # Update selection rect.


def fsetToMax(f_set, n, maxIndex, maxIndexToSend):
    global verSlider
    if not f_set.is_set():
        if n > maxIndex:
            f_set.set()
            f_restToMin = threading.Event()
            fresetToMin(f_restToMin, verSlider.get(), maxIndexToSend)
        else:
            verSlider.set(n)
            threading.Timer(stepDelay, fsetToMax, [f_set, (n + 1), maxIndex, maxIndexToSend]).start()


def fresetToMin(f_rest, n, maxIndex):
    global verSlider
    if not f_rest.is_set():
        if n < 0:
            f_rest.set()
            f_setToBest = threading.Event()
            fsetToBest(f_setToBest, verSlider.get(), maxIndex)
        else:
            verSlider.set(n)
            threading.Timer(stepDelay, fresetToMin, [f_rest, (n - 1), maxIndex]).start()


def fsetToBest(f_set, n, maxIndex):
    global verSlider
    if not f_set.is_set():
        if n > maxIndex:
            f_set.set()
            verSlider.set(int(n - 0.5))
        else:
            verSlider.set(n)
            threading.Timer(stepDelay, fsetToBest, [f_set, (n + 1), maxIndex]).start()


def windowCoordinatesCallback(event):
    global globXclick, globYclick, prevGlobXclick, prevGlobYclick, prevGlobXclick, prevGlobYclick, regionSelection, status2
    if regionSelection == 'Manual':
        if prevGlobXclick == globXclick and prevGlobYclick == globYclick:
            status2.config(text='Please select the region within image frame...')
        else:
            status2.config(text='')
    prevGlobXclick, prevGlobYclick = globXclick, globYclick


def imageCoordinatesCallback(event):
    global globXclick, globYclick, regionSelection, sharpnessMeasure, patchSizeXx, patchSizeYy, status2, img
    globXclick, globYclick = event.x, event.y


def setValues():
    global scene, AFobjective, sharpnessMeasure, regionSelection, kernelSize, patchSizeXx, patchSizeYy, patchSizeRatio, defScene, defObjective, defSharpness, defRegionSel, defKerSel, defPatchRatioSel, img, status1, status2, img_all, numOfFrames
    if scene != defScene.get():
        img_all = np.load(defScene.get() + '_imgs_' + str(downscale_ratio) + '.npy')
        if defScene.get() == 'Scene1' or defScene.get() == 'Scene2' or defScene.get() == 'Scene3' or defScene.get() == 'Scene4':
            numOfFrames = 51
        elif defScene.get() == 'Scene5':
            numOfFrames = 61
        elif defScene.get() == 'Scene6' or defScene.get() == 'Scene7' or defScene.get() == 'Scene8':
            numOfFrames = 71
        elif defScene.get() == 'Scene9' or defScene.get() == 'Scene10':
            numOfFrames = 91
        resetValues()
        scene = defScene.get()
        status1.config(text='AF Data Browser Ready...')
        status2.config(
            text='Default: ' + scene + ', ' + AFobjective + ' objective, ' + sharpnessMeasure + ' sharpness measure, ' + regionSelection + ' region, and kernel size = ' + str(
                kernelSize) + ' are selected')


def resetValues():
    global verSlider, horSlider, photo, defScene, defObjective, defSharpness, defRegionSel, defKerSel, defPatchRatioSel, AFobjective, sharpnessMeasure, regionSelection, kernelSize, patchSizeRatio, patchSizeXx, patchSizeYy, img, numOfFrames
    defObjective.set('Global'), defSharpness.set('Sobel'), defRegionSel.set('Auto'), defKerSel.set(
        '3'), defPatchRatioSel.set(str(0.0625))
    AFobjective, sharpnessMeasure, regionSelection, kernelSize, patchSizeRatio = defObjective.get(), defSharpness.get(), defRegionSel.get(), int(
        defKerSel.get()), float(defPatchRatioSel.get())
    patchSizeXx, patchSizeYy = int(patchSizeRatio * img.shape[1]), int(patchSizeRatio * img.shape[0])
    horSlider.config(to=numOfFrames - 1)
    verSlider.set(0), horSlider.set(0)
    tempImg = Image.fromarray(img_all[0])
    photo = ImageTk.PhotoImage(tempImg)
    imagePrevlabel.configure(image=photo)


def updateSlider(event):
    global verSlider, horSlider, photo, photo_2, imagePrevlabel, imagePrevlabel_2, img_all, img, img_mean_list, stack_size, fig
    temp_img_ind = int(horSlider.get()) * stack_size + int(verSlider.get())
    img = deepcopy(img_all[temp_img_ind])
    tempImg = Image.fromarray(img)
    photo = ImageTk.PhotoImage(tempImg)
    imagePrevlabel.configure(image=photo)
    # Image mean plot
    plt.close(fig)
    fig.clear()
    fig = plt.figure(figsize=(4.6, 3.6))
    plt.plot(np.arange(stack_size), img_mean_list[(temp_img_ind // stack_size) * stack_size:(
                                                                                                        temp_img_ind // stack_size) * stack_size + stack_size],
             color='green', linewidth=2)
    plt.plot(int(verSlider.get()), img_mean_list[temp_img_ind], color='red', marker='o', markersize=12)
    plt.text(int(verSlider.get()), img_mean_list[temp_img_ind],
             '(' + str(int(verSlider.get())) + ', ' + str("%.2f" % img_mean_list[temp_img_ind]) + ')', color='red',
             fontsize=13, position=(verSlider.get() - 0.2, img_mean_list[temp_img_ind] + 0.04))
    plt.title('Exposure stack mean')
    plt.xlabel('Image index')
    plt.ylabel('Mean value')
    plt.xlim(-0.2, stack_size - 0.8)
    plt.xticks(np.arange(0, stack_size, 1))
    plt.ylim(-0.02, 0.85)
    plt.yticks(np.arange(0, 0.85, 0.1))
    fig.canvas.draw()

    tempImg_2 = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    photo_2 = ImageTk.PhotoImage(tempImg_2)
    imagePrevlabel_2.configure(image=photo_2)


################################################################################Main()
widgetFont, widgetFontSize, drawThickness, focusPoints, pltWinW, pltWinH, stepDelay, patchSizeRatio = 'DejaVu Sans', 14, 3, 0, 19, 9.5, 0.005, 0.0625
globFaceLocations, globLargestFaseAreaInd = 0, 0
scene, AFobjective, sharpnessMeasure, regionSelection, kernelSize = 'Scene1', 'Global', 'Sobel', 'Auto', 3
topFaceReg, leftFaceReg, bottomFaceReg, rightFaceReg = 0, 0, 0, 0
globXclick, globYclick, prevGlobXclick, prevGlobYclick, globMaxFit, globMaxInd, globMaxFPInd, patchSelectedFlag = 0, 0, 0, 0, 0, 0, 0, False
prewittKernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewittKernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
###############################
if scene == 'Scene1' or scene == 'Scene2' or scene == 'Scene3' or scene == 'Scene4':
    numOfFrames = 90

bit_depth = 8
stack_size = 12

downscale_ratio = 0.12

imgSize = [int(4480 * downscale_ratio), int(6720 * downscale_ratio)]
widthToScale = imgSize[1]
widPercent = (widthToScale / float(imgSize[1]))
heightToScale = int(float(imgSize[0]) * float(widPercent))

img_all = np.load(scene + '_imgs_' + str(downscale_ratio) + '.npy')
img_mean_list = np.load(scene + '_img_mean_' + str(downscale_ratio) + '.npy') / (2 ** bit_depth - 1)
im_all_quarter = []
for im in img_all:
    im_quarter_temp = cv2.resize(im, None, fx=0.25, fy=0.25)
    im_all_quarter.append(im_quarter_temp)

tempImg, img = deepcopy(img_all[0]), deepcopy(img_all[0])
tempImg2, img2 = deepcopy(im_all_quarter[7]), deepcopy(im_all_quarter[7])
patchSizeXx, patchSizeYy = int(patchSizeRatio * tempImg.shape[1]), int(patchSizeRatio * tempImg.shape[0])
#################################

# Tkinter Window
root = tk.Tk()
root.geometry('1900x1000+5+5'), root.title('Data Browser'), root.iconbitmap('AF_Icon.ico')
root.bind('<Button-1>', windowCoordinatesCallback)


tempImg = Image.fromarray(tempImg)
tempImg2 = Image.fromarray(tempImg2)
photo = ImageTk.PhotoImage(tempImg)
photo2 = ImageTk.PhotoImage(tempImg2)
print("size{}X{}".format(photo2.height(), photo2.width()))

imagePrevlabel = tk.Label(root, image=photo)
imagePrevlabel.grid(row=1, column=1, rowspan=30, sticky=tk.NW)
imagePrevlabel.bind('<Button-1>', imageCoordinatesCallback)

canvas = tk.Canvas(root, width=photo2.width(), height=photo2.height(),
                   borderwidth=0, highlightthickness=0)
canvas.grid(row=3, column=2, rowspan=30,columnspan=2, sticky=tk.NW)
canvas.img = photo  # Keep reference in case this code is put into a function.
canvas.create_image(0, 0, image=photo2, anchor=tk.NW)

# Create selection rectangle (invisible since corner points are equal).
rect_id = canvas.create_rectangle(topx, topy, topx, topy,
                                  dash=(2, 2), fill='blue', outline='blue')

canvas.bind('<Button-1>', get_mouse_posn)

# Image mean plot
fig = plt.figure(figsize=(4.6, 3.6))
plt.plot(np.arange(stack_size), img_mean_list[0:stack_size], color='green',
         linewidth=2)  # ,label='Exposure stack mean')
plt.plot(0, img_mean_list[0], color='red', marker='o', markersize=12)
plt.text(0, img_mean_list[0], '(' + str(0) + ', ' + str("%.2f" % img_mean_list[0]) + ')', color='red', fontsize=13,
         position=(0 - 0.2, img_mean_list[0] + 0.04))
plt.title('Exposure stack mean')
plt.xlabel('Image index')
plt.ylabel('Mean value')
plt.xlim(-0.2, stack_size - 0.8)
plt.xticks(np.arange(0, stack_size, 1))
plt.ylim(-0.02, 0.85)
plt.yticks(np.arange(0, 0.85, 0.1))
fig.canvas.draw()

tempImg_2 = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
photo_2 = ImageTk.PhotoImage(tempImg_2)
imagePrevlabel_2 = tk.Label(root, image=photo_2)
# imagePrevlabel_2.grid(row=5, column=2,  columnspan=2, rowspan=24, sticky=tk.NE)

# Vertical Slider
verSliderLabel = tk.Label(root, text='Exposure Time', font=(widgetFont, widgetFontSize))
verSliderLabel.grid(row=0, column=0)
verSlider = tk.Scale(root, activebackground='black', cursor='sb_v_double_arrow', from_=0, to=stack_size - 1,
                     font=(widgetFont, widgetFontSize), length=heightToScale, command=updateSlider)
verSlider.grid(row=1, column=0, rowspan=30)

# Horizantal Slider
horSlider = tk.Scale(root, activebackground='black', cursor='sb_h_double_arrow', from_=0, to=numOfFrames - 1,
                     label='Frame Number', font=(widgetFont, widgetFontSize), orient=tk.HORIZONTAL, length=widthToScale,
                     command=updateSlider)
horSlider.grid(row=31, column=1, sticky=tk.SW)


#calculate the total variation of an 8*8 matrix
def getTV(imm):
    im = imm
    im = np.array(im, dtype=np.uint8).astype(np.int16, casting='same_kind')
    if boty < im.shape[0] - 1:
        hvariance = np.absolute(im[topy:boty, topx:botx, 1] - im[topy + 1:boty + 1, topx:botx, 1])
    else:
        w = np.zeros((1, botx - topx + 1))
        hvariance_temp = np.absolute(im[topy:boty - 1, topx:botx, 1] - im[topy + 1:boty, topx:botx, 1])
        hvariance = np.concatenate((hvariance_temp, w), axis=0)
    if botx < im.shape[1] - 1:
        vvariance = np.absolute(im[topy:boty, topx:botx, 1] - im[topy:boty, topx + 1:botx + 1, 1])
    else:
        w = np.zeros(boty - topy + 1)
        vvariance_temp = np.absolute(im[topy:boty, topx:botx - 1, 1] - im[topy:boty, topx + 1:botx, 1])
        vvariance = np.concatenate((vvariance_temp, w), axis=0)
    total = vvariance + hvariance
    cnonzero = np.count_nonzero(total)
    t = np.sum(total)
    return t, cnonzero



#calculate the total variation of the stack of 8*8 matrices
def get_all_TV():
    x = []
    stack_n = 0
    max_t = 0
    max_t_in = 0
    for i, ims in enumerate(im_all_quarter):
        if int(i / 12) > stack_n:
            x.append(max_t_in)
            max_t = 0
            max_t_in = 0
            stack_n += 1
        t, cnonzero = getTV(ims)
        if t >= max_t:
            max_t_in = i % 12
            max_t = max(max_t, t)
        #print("{},{},{},{}".format(i % 12, int(i / 12), t, cnonzero))
    x.append(max_t_in)
    return (x)


def set_target():
    global opt
    global ii
    opt = get_all_TV()
    horSlider.set(0)
    verSlider.set(0)
    ii = 0
    b_forward = tk.Button(root, text=">>", command=lambda: forward_play(opt))
    b_forward.grid(row=2, column=3)
    b_back = tk.Button(root, text='<<', state=tk.DISABLED)
    b_back.grid(row=2, column=2)


def forward_play(opt):

    #global opt
    global ii
    if ii == len(opt):
        ii = ii - 2
        b_forward = tk.Button(root, text=">>", state=tk.DISABLED)
        b_forward.grid(row=2, column=3)
        return
    horSlider.set(ii)
    verSlider.set(opt[ii])
    ii += 1
    if ii == 1 or ii == 2:
        b_back = tk.Button(root, text='<<', command=lambda:back_play(opt))
        b_back.grid(row=2, column=2)


def back_play(opt):
    global ii
    if ii == -1:
        ii = 1
        b_back = tk.Button(root, text='<<', state=tk.DISABLED)
        b_back.grid(row=2, column=2)
        return
    horSlider.set(ii)
    verSlider.set(opt[ii])
    ii -= 1
    if ii == len(opt)-2 or ii == len(opt) - 3:
        b_forward = tk.Button(root, text=">>", command=lambda: forward_play(opt))
        b_forward.grid(row=2, column=3)


b = tk.Button(root, text="Set target", command=set_target)
b.grid(row=1, column=2,columnspan=2)
b_forward = tk.Button(root,text=">>",command=lambda: forward_play(opt))
b_forward.grid(row=2,column=3)
b_back = tk.Button(root,text='<<',state=tk.DISABLED)
b_back.grid(row=2,column=2)
# Select Scene List
# defScene = tk.StringVar(root)
# defScene.set('Scene1') # default value
# selSceneLabel=tk.Label(root, text='Select Scene:', font=(widgetFont, widgetFontSize))
# selSceneLabel.grid(row=0, column=2, sticky=tk.W)
# sceneList = tk.OptionMenu(root, defScene, 'Scene1')#, 'Scene2', 'Scene3', 'Scene4', 'Scene5', 'Scene6', 'Scene7', 'Scene8', 'Scene9', 'Scene10')
# sceneList.config(font=(widgetFont, widgetFontSize-2), width=15, anchor=tk.W)
# sceneList.grid(row=1, column=2, sticky=tk.NE)

# #Run Button
# RunButton=tk.Button(root, text='Run', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=18, font=(widgetFont, widgetFontSize), command=setValues)
# RunButton.grid(row=12, column=2, sticky=tk.E)

# #Reset Button
# RunButton=tk.Button(root, text='Reset', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=18, font=(widgetFont, widgetFontSize), command=resetValues)
# RunButton.grid(row=31, column=2, sticky=tk.E)

# #Home Button
# HomeButton=tk.Button(root, text='Home', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=18, font=(widgetFont, widgetFontSize))
# HomeButton.grid(row=32, column=2, rowspan=2, sticky=tk.E)

# #Status bar
# status1=tk.Label(root, text='AF Data Browser Ready...', font=(widgetFont, widgetFontSize-2), bd=1, relief=tk.SUNKEN, anchor=tk.W)
# #status1.grid(row=32, column=0, columnspan=2, sticky=tk.SW)
# #Status bar
# status2=tk.Label(root, text='Default: '+scene+', '+AFobjective+' objective, '+sharpnessMeasure+' sharpness measure, '+regionSelection+' region, and kernel size = '+str(kernelSize)+' are selected', font=(widgetFont, widgetFontSize-2), bd=1, relief=tk.SUNKEN, anchor=tk.W)
# status2.grid(row=33, column=0, columnspan=2, sticky=tk.SW)

root.mainloop()
