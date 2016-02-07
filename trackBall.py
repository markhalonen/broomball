__author__ = 'brent'
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import ntpath
import tkinter as tk
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

def displayImage(img, windowName = "Image"):

    imHeight = img.shape[0]
    imWidth = img.shape[1]
    if imWidth / imHeight > screen_width / screen_height:
        temp = cv2.resize(img,(screen_width * 2/ 3,screen_width * 2 / 3 * imHeight / imWidth))
    else:
        temp = cv2.resize(img,(screen_height * 2 / 3 * imWidth / imHeight,screen_height * 2/ 3))
    cv2.imshow(windowName,temp)
    cv2.waitKey(0)

imagesDir = "/Users/Mark/Google Drive/BroomballAlgo/images"
maskDir = "/Users/Mark/Google Drive/BroomballAlgo/silverRinkMask.png"
mask = cv2.imread(maskDir, 0)

for file in os.listdir(imagesDir):
    im = cv2.imread(imagesDir + "/" + file,0)
    height, width = im.shape[:2]
    mask = cv2.resize(mask, (width, height))
    res = cv2.bitwise_and(im,im,mask = mask)

    ret2,thres = cv2.threshold(res,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret2,thres = cv2.threshold(res,0,255,204)
    displayImage(thres)
    #plt.imshow(thres),plt.show()

