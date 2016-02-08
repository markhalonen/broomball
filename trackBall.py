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


pts = [(803, 152), (803, 152), (1116, 181), (880, 172), (470, 180), (885, 177), (472, 181), (920, 124), (810, 118), (811, 118), (897, 190), (813, 117), (486, 188), (904, 197), (907, 200), (951, 352), (944, 343), (562, 456), (949, 337), (821, 117), (822, 117), (923, 217), (926, 219), (826, 117), (827, 117), (933, 226), (562, 404), (993, 323), (994, 321), (942, 235), (944, 237), (859, 95), (942, 236), (941, 236), (940, 236), (820, 303), (936, 237), (1116, 180), (987, 279), (568, 388), (1117, 180), (1011, 282), (570, 384), (922, 243), (752, 151), (976, 262), (573, 379), (1116, 180), (989, 255), (576, 375), (911, 247), (577, 373), (908, 248), (578, 371), (906, 249), (975, 276), (578, 368), (578, 367), (925, 245), (577, 365), (493, 234), (576, 363), (575, 362), (575, 361), (574, 360), (897, 254), (574, 358), (574, 357), (895, 255), (574, 356), (574, 356), (574, 355), (574, 355), (573, 354), (573, 354), (572, 353), (571, 353), (571, 352), (570, 352), (880, 261), (569, 351), (877, 262), (876, 263), (855, 290), (873, 264), (566, 350), (870, 265), (565, 349), (867, 265), (564, 349), (563, 348), (563, 348), (835, 258), (562, 347), (819, 256), (562, 347), (805, 253), (562, 347), (562, 347), (562, 347), (514, 245), (562, 346), (561, 346), (561, 346), (561, 346), (745, 242), (739, 241), (858, 213), (728, 239), (557, 346), (858, 214), (858, 214), (553, 345), (816, 275), (860, 213), (509, 251), (547, 344), (862, 213), (862, 213), (863, 213), (863, 213), (506, 254), (680, 240), (864, 212), (534, 340), (675, 243), (532, 339), (531, 339), (530, 339), (529, 339), (529, 339), (500, 274), (868, 211), (986, 187), (986, 187), (523, 337), (1010, 210), (520, 336), (868, 213), (868, 213), (867, 213), (867, 213), (866, 214), (866, 214), (866, 214), (866, 214), (865, 214), (532, 381), (652, 266), (865, 213), (651, 268), (650, 269), (509, 331), (508, 330), (1251, 233), (525, 378), (647, 273), (1251, 233), (1251, 233), (645, 276), (645, 277), (644, 278), (644, 279), (1251, 233), (643, 280), (642, 281), (642, 282), (641, 283), (670, 320), (841, 233), (639, 285), (639, 286), (638, 287), (638, 288), (253, 438), (637, 289), (636, 290), (636, 291), (1251, 233), (1251, 233), (634, 293), (495, 317), (615, 304), (603, 311), (591, 318), (579, 326), (567, 334), (555, 342), (895, 268), (530, 357), (1251, 233), (506, 372), (493, 380), (480, 388), (497, 320), (758, 117), (442, 412), (497, 319), (496, 319), (495, 318), (494, 317), (493, 317), (492, 315), (491, 314), (816, 239), (489, 313), (488, 312), (902, 206), (812, 240), (487, 312), (810, 241), (810, 241), (809, 241), (808, 241), (484, 309), (245, 435), (482, 310), (499, 390), (806, 242), (489, 332), (478, 318), (884, 252), (488, 332), (500, 402), (487, 332), (868, 224), (486, 331), (728, 153), (857, 263), (286, 588), (794, 248), (909, 273), (909, 273), (596, 294), (852, 234), (479, 329), (786, 258), (785, 260), (814, 278), (476, 333), (312, 617), (476, 339), (476, 342), (318, 625), (475, 348), (396, 395), (391, 401), (214, 448), (1251, 233), (470, 361), (386, 406), (830, 247), (1251, 233), (335, 650), (441, 384), (338, 656), (820, 254), (1251, 233), (815, 256), (776, 278), (362, 489), (1251, 233), (813, 301), (1251, 233), (1251, 233), (427, 396), (426, 397), (425, 398), (423, 398), (422, 399), (247, 440), (362, 696), (247, 439), (1251, 233), (310, 458), (1116, 180), (1116, 180), (780, 284), (1116, 180), (796, 311), (409, 417), (377, 721), (379, 723), (409, 419), (794, 315), (323, 580), (1251, 233), (1251, 233), (600, 353), (600, 353), (421, 620), (789, 320), (788, 321), (787, 322), (735, 319), (735, 315), (424, 639), (393, 416), (1251, 233), (703, 309), (717, 328), (765, 342), (707, 355), (707, 355), (779, 328), (608, 394), (608, 394), (608, 394), (419, 727), (694, 359), (670, 322), (424, 723), (665, 325), (737, 365), (335, 715), (1251, 233), (616, 421), (1251, 233), (1251, 233), (659, 390), (614, 437), (1251, 233), (1251, 233), (1251, 233), (661, 382), (417, 411), (1251, 233), (1251, 233), (616, 450), (1251, 233), (617, 452), (1251, 233), (730, 324), (619, 457), (399, 729), (726, 326), (620, 460), (620, 461), (287, 558), (719, 327), (717, 327), (395, 730), (395, 730), (394, 730), (1251, 233), (1251, 233), (706, 328), (692, 341), (703, 329), (239, 479), (700, 330), (699, 330), (697, 331), (397, 712), (398, 710), (398, 708), (620, 464), (1108, 186), (357, 686), (405, 730), (367, 431), (368, 431), (346, 566), (418, 649), (428, 633), (438, 618), (1108, 185), (1108, 185), (1108, 185), (1108, 185), (1108, 185), (1108, 185), (1108, 185), (367, 430), (1108, 185), (1108, 186), (365, 429), (521, 534), (462, 518), (530, 542), (363, 430), (654, 340), (546, 556), (551, 561), (555, 564), (559, 567), (652, 340), (568, 574), (573, 578), (464, 564), (384, 409), (587, 589), (591, 592), (496, 620), (600, 598), (604, 601), (608, 604), (613, 607), (617, 610), (621, 613), (608, 474), (630, 619), (221, 423), (608, 473), (642, 627), (221, 423), (653, 345), (655, 345), (657, 345), (659, 345), (453, 551), (370, 410), (370, 410), (667, 344), (370, 411), (670, 344), (671, 343), (673, 343), (674, 343), (616, 488), (465, 552), (702, 196), (372, 412), (372, 412), (707, 195), (373, 412), (710, 620), (374, 412), (379, 432), (694, 343), (614, 490), (616, 492), (378, 412), (379, 412), (380, 412), (703, 337), (704, 336), (706, 335), (708, 334), (491, 301), (734, 204), (714, 331), (718, 606), (718, 329), (690, 452), (394, 430), (743, 200), (745, 177), (664, 518), (245, 473), (749, 175), (683, 525), (831, 584), (755, 579), (1251, 233), (730, 533), (805, 568), (745, 532), (1251, 233), (865, 557), (870, 553), (875, 549), (879, 545), (884, 542), (842, 551), (892, 535), (773, 309), (900, 528), (762, 537), (909, 522), (787, 539), (783, 304), (786, 303), (823, 536), (928, 507), (828, 535), (797, 298), (410, 586), (802, 297), (805, 296), (823, 532), (1108, 186), (812, 294), (814, 293), (816, 291), (229, 440), (970, 478), (459, 365), (976, 474), (228, 442), (829, 314), (472, 414), (831, 313), (838, 276), (840, 275), (843, 274), (938, 492), (848, 272), (851, 271), (854, 270), (856, 269), (858, 268), (860, 266), (1107, 184), (490, 407), (499, 408), (870, 259), (978, 510), (649, 265), (878, 253), (1019, 372), (882, 250), (884, 249), (991, 417), (887, 247), (889, 246), (890, 246), (892, 245), (233, 445), (859, 159), (896, 245), (897, 244), (999, 360), (765, 414), (1108, 185), (235, 442), (235, 442), (906, 236), (234, 442), (1108, 186), (233, 442), (913, 229), (523, 344), (232, 442), (232, 442), (232, 442), (232, 442), (232, 442), (232, 442), (924, 219), (1057, 296), (1057, 294), (1057, 292), (1063, 200), (1058, 289), (1058, 288), (800, 215), (1061, 284), (236, 442), (236, 442), (786, 408), (788, 407), (237, 442), (939, 202), (940, 201), (238, 442), (238, 442), (942, 198), (239, 441), (239, 441), (239, 441), (560, 320), (852, 385)]


angleThreshold = 20 * math.pi / 180
ptsInARowNecessary = 3
prevPoints = []
ptsWithIndex = []
ptsAngle = []
index = 0
for (x,y) in pts:

    #for the beginning of the array
    if len(prevPoints) < ptsInARowNecessary:
        prevPoints.append((x,y))
        continue
    passes = True
    prevX = None
    prevY = None
    prevX2 = None
    prevY2 = None
    for (x1, y1) in prevPoints:
        if prevX is None :
            prevX = x1
            prevY = y1
            continue
        if prevX2 is None:
            prevX2 = x1
            prevY2 = y1
            continue
        ang1 = cartesian2polar((prevX - x1), (prevY - y1))[1]
        ang2 = cartesian2polar((prevX2 - prevX), (prevY2 - prevY2))[1]
        diff = min((2 * math.pi) - abs(ang1 - ang2), abs(ang1 - ang2))
        if diff > angleThreshold:
            passes = False
            break
        prevX = prevX2
        prevY = prevY2
        prevX2 = x1
        prevY2 = y1

    if passes:
        ptsWithIndex.append(((x,y), index))
        ptsAngle.append((x,y))
    # firstAngle = cartesian2polar((x2 - x1), (y2 - y1))[0]
    # secondAngle = cartesian2polar((x3 - x2), (y3 - y2))[0]
    # angleDiff = min((2 * math.pi) - abs(firstAngle - secondAngle), abs(firstAngle - secondAngle))
    # if(angleDiff < angleThreshold):
    #     ptsWithIndex.append(((x3,y3),index))
    #     ptsAngle.append((x3,y3))
    #
    # x1 = x2
    # y1 = y2
    # x2 = x3
    # y2 = y3

    prevPoints = prevPoints[:len(prevPoints) - 1]
    prevPoints.append((x,y))

    index = index + 1

#plt.plot(ptsAngle, 'ro')

plt.plot(ptsAngle, 'ro')
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

plt.plot(pts, 'ro')
plt.axis()
plt.show()

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