__author__ = 'brent'
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import math
import ntpath
import imutils
import matplotlib.pyplot as plt
from sympy import Line, Point, Polygon
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

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def cartesian2polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def polar2cartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


imagesDir = "images"
maskDir = "silverRinkMask.png"
ballMaskDir = "ballMask.png"
ballMaskDir2 = "ballMask2.png"
mask = cv2.imread(maskDir, 0)
ballMask = cv2.imread(ballMaskDir, 0)
ballMask2 = cv2.imread(ballMaskDir2, 0)
ballSizeInches = 5
inchToPixelRatio = .1
index = 0


pts = [(803, 152), (803, 152), (1116, 181), (880, 172), (470, 180), (885, 177), (472, 181), (920, 124), (810, 118), (811, 118), (897, 190), (813, 117), (486, 188), (904, 197), (907, 200), (951, 352), (944, 343), (562, 456), (949, 337), (821, 117), (822, 117), (923, 217), (926, 219), (826, 117), (827, 117), (933, 226), (562, 404), (993, 323), (994, 321), (942, 235), (944, 237), (859, 95), (942, 236), (941, 236), (940, 236), (820, 303), (936, 237), (1116, 180), (987, 279), (568, 388), (1117, 180), (1011, 282), (570, 384), (922, 243), (752, 151), (976, 262), (573, 379), (1116, 180), (989, 255), (576, 375), (911, 247), (577, 373), (908, 248), (578, 371), (906, 249), (975, 276), (578, 368), (578, 367), (925, 245), (577, 365), (493, 234), (576, 363), (575, 362), (575, 361), (574, 360), (897, 254), (574, 358), (574, 357)]

angleThreshold = 20 * math.pi / 180
x1 = None
y1 = None
x2 = None
y2 = None
ptsWithIndex = []
ptsAngle = []
index = 0
for (x3,y3) in pts:
    #for the beginning of the array
    if x1 is None:
        x1 = x3
        y1 = y3
        continue
    if x2 is None:
        x2 = x3
        y2 = y3

    firstAngle = cartesian2polar((x2 - x1), (y2 - y1))[0]
    secondAngle = cartesian2polar((x3 - x2), (y3 - y2))[0]
    angleDiff = min((2 * math.pi) - abs(firstAngle - secondAngle), abs(firstAngle - secondAngle))
    if(angleDiff < angleThreshold):
        ptsWithIndex.append(((x3,y3),index))
        ptsAngle.append((x3,y3))

    x1 = x2
    y1 = y2
    x2 = x3
    y2 = y3

    index = index + 1

#plt.plot(ptsAngle, 'ro')
plt.plot(pts, 'ro')
plt.axis()
plt.show()


print ptsWithIndex

imageIndex = 0
keptPointsIndex = 0
for file in os.listdir(imagesDir):
    if index < 133:
        index = index + 1
        continue
    colorIm = cv2.imread(imagesDir + "/" + file)
    print "Image Index: " + str(imageIndex)
    print "Pts index: " + str(ptsWithIndex[keptPointsIndex][1])
    if imageIndex == ptsWithIndex[keptPointsIndex][1]:
        cop = colorIm.copy()
        #cv2.circle(circ,(256,256), r, (255,255,255), -1)
        cv2.circle(cop, (ptsWithIndex[keptPointsIndex][0][0], ptsWithIndex[keptPointsIndex][0][1]), 10, (255,0,0), -1)
        displayImage(cop)
        keptPointsIndex += 1
    imageIndex += 1












pts = []
for file in os.listdir(imagesDir):
    if index < 133:
        index = index + 1
        continue
    if index > 200:
        break
    colorIm = cv2.imread(imagesDir + "/" + file)
    im = cv2.imread(imagesDir + "/" + file,0)
    im = cv2.GaussianBlur(im,(5,5),0)
    height, width = im.shape[:2]
    mask = cv2.resize(mask, (width, height))
    res = cv2.bitwise_and(im,im,mask = mask)

    ret2,thres = cv2.threshold(res,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    (cnts, _) = cv2.findContours(thres.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

    print file
    print index
    print str(len(cnts)) + " :num contours"

    #loop over the contours
    leastDiff = 10000000000
    for c in cnts:
        # if the contour is too large, ignore it
        if cv2.contourArea(c) > 150 or cv2.contourArea(c) < 10:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text.

        (x, y, w, h) = cv2.boundingRect(c)


        # compare each contour to the ball mask
        subRect = thres[y:(y + w), x:(x + w)]
        (h1, w1) = subRect.shape[:2]
        #the circle has different padding around it
        for r in xrange(200, 256,5):
            circ = np.zeros((512,512), np.uint8)
            cv2.circle(circ,(256,256), r, (255,255,255), -1)
            rev = 255 - circ
            rev = cv2.resize(rev, (w1, h1))
            diff = mse(subRect, rev)
            if(diff < leastDiff):
                leastDiff = diff
                winningRect = (x,y,w,h)
            #displayImage(rev)



        # ballMaskResized = cv2.resize(ballMask, (w, h))
        # ballMaskResized2 = cv2.resize(ballMask2, (w, h))
        # diff = mse(subRect, ballMaskResized)
        # diff2 = mse(subRect, ballMaskResized2)
        # print "Diff: " + str(diff)
        # if(diff < leastDiff):
        #     leastDiff = diff
        #     winningRect = (x,y,w,h)
        # if(diff2 < leastDiff):
        #     leastDiff = diff2
        #     winningRect = (x,y,w,h)

        # cop = colorIm.copy()
        # cv2.rectangle(cop, (x, y), (x + w, y + h), (0, 255, 0), 5)
        # displayImage(cop)
        #
        # displayImage(subRect)

    (x,y,w,h) = winningRect
    pts.append((x + w/2, y + h/2))
    colored = colorIm.copy()
    cv2.rectangle(colored, (x, y), (x + w, y + h), (0, 255, 0), 5)
    sub = colorIm[y:(y + w), x:(x + w)]
    #ret2,thres = cv2.threshold(res,0,255,204)
    # displayImage(colored)
    # displayImage(sub)
    index = index + 1
    #plt.imshow(thres),plt.show()

print pts

#go through pts, if 3 points go in the same direction, keep the points and the index, see if we can track the ball
angleThreshold = 20 * math.pi / 180
x1 = None
y1 = None
x2 = None
y2 = None
ptsWithIndex = []
index = 0
for (x3,y3) in pts:
    #for the beginning of the array
    if x1 is None:
        x1 = x3
        y1 = y3
        continue
    if x2 is None:
        x2 = x3
        y2 = y3

    firstAngle = cartesian2polar((x2 - x1), (y2 - y1))[0]
    secondAngle = cartesian2polar((x3 - x2), (y3 - y2))[0]
    angleDiff = min((2 * math.pi) - abs(firstAngle - secondAngle), abs(firstAngle - secondAngle))
    if(angleDiff < angleThreshold):
        ptsWithIndex.append(((x3,y3),index))

    x1 = x2
    y1 = y2
    x2 = x3
    y2 = y3

    index = index + 1

print ptsWithIndex

imageIndex = 0
keptPointsIndex = 0
for file in os.listdir(imagesDir):
    if index < 133:
        index = index + 1
        continue
    colorIm = cv2.imread(imagesDir + "/" + file)
    print "Image Index: " + str(imageIndex)
    print "Pts index: " + ptsWithIndex[keptPointsIndex][1]
    if imageIndex == ptsWithIndex[keptPointsIndex][1]:
        cop = colorIm.copy()
        #cv2.circle(circ,(256,256), r, (255,255,255), -1)
        cv2.circle(cop, (ptsWithIndex[keptPointsIndex][0][0], ptsWithIndex[keptPointsIndex][0][1]), 10, (255,0,0), -1)
        displayImage(cop)
        keptPointsIndex += 1
    imageIndex += 1