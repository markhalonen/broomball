__author__ = 'brent'
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import ntpath
import imutils
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

imagesDir = "images"
maskDir = "silverRinkMask.png"
mask = cv2.imread(maskDir, 0)
ballSizeInches = 5
inchToPixelRatio = .1

for file in os.listdir(imagesDir):
    colorIm = cv2.imread(imagesDir + "/" + file)
    im = cv2.imread(imagesDir + "/" + file,0)
    height, width = im.shape[:2]
    mask = cv2.resize(mask, (width, height))
    res = cv2.bitwise_and(im,im,mask = mask)

    ret2,thres = cv2.threshold(res,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    (cnts, _) = cv2.findContours(thres.copy(), cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE)

    print str(len(cnts)) + " :num contours"
    #loop over the contours
    for c in cnts:
		# if the contour is too large, ignore it
		if cv2.contourArea(c) > 150 or cv2.contourArea(c) < 10:      
		 	continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(colorIm, (x, y), (x + w, y + h), (0, 255, 0), 5)


    #ret2,thres = cv2.threshold(res,0,255,204)
    displayImage(colorIm)
    #plt.imshow(thres),plt.show()

