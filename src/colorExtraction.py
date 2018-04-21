#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: colorExtraction.py
Author: Cade Parkison
Email: cadeparkison@gmail.com
Github: c-park
Description:
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

from faceDetection import recognizeCube
from plotting import plotCube

# Color Bounds
# R_lower = np.array([-10, 100, 100])
# R_upper = np.array([10, 255, 255])

# G_lower = np.array([61, 100, 100])
# G_upper = np.array([81, 255, 255])

# Y_lower = np.array([19, 100, 100])
# Y_upper = np.array([39, 255, 255])

# O_lower = np.array([2, 100, 100])
# O_upper = np.array([22, 255, 255])

# B_lower = np.array([102, 100, 100])
# B_upper = np.array([122, 255, 255])

# W_lower = np.array([41, 100, 100])
# W_upper = np.array([61, 255, 255])

W_lower,W_upper = np.array([0,0,196]), np.array([179,21,255])
R_lower,R_upper = np.array([0,100,100]), np.array([179,255,255])
Y_lower,Y_upper = np.array([23,100,100]), np.array([39,255,255])
O_lower,O_upper = np.array([7,100,100]), np.array([28,255,255])
B_lower,B_upper = np.array([94,100,100]), np.array([127,255,255])
G_lower,G_upper = np.array([60,29,100]), np.array([80,255,255])

W_lower,W_upper = np.array([0,0,245]), np.array([6,23,255])
W_lower2,W_upper2 = np.array([104,0,188]), np.array([179,23,255])
R_lower,R_upper = np.array([0,137,145]), np.array([6,255,255])
Y_lower,Y_upper = np.array([26,137,108]), np.array([42,255,255])
O_lower,O_upper = np.array([11,136,118]), np.array([21,255,255])
B_lower,B_upper = np.array([106,145,101]), np.array([114,255,255])
G_lower,G_upper = np.array([39, 114, 96]), np.array([98,255,255])

# Red lower(0,137,145) upper(6,255,255)
# Yellow lower(26,137,108) upper(42,255,255)
# Orange lower(11,136,118) upper(21,255,255)
# Green lower(39,114,96) upper(98,255,255)
# Blue lower(106,145,101) upper(114,255,255)
# White lower(0,0,245) upper (6,23,255) and lower(104,0,188) upper(179,23,255)


# color = avgColor(hsv, c, boxHeight, boxWidth)

def getColor(hsv):
    if (W_lower <= hsv).all() & (W_upper > hsv).all(): return 'White'
    elif (R_lower <= hsv).all() & (R_upper > hsv).all(): return 'Red'
    elif (Y_lower <= hsv).all() & (Y_upper > hsv).all(): return 'Yellow'
    elif (O_lower <= hsv).all() & (O_upper > hsv).all(): return 'Orange'
    elif (B_lower <= hsv).all() & (B_upper > hsv).all(): return 'Blue'
    elif (G_lower <= hsv).all() & (G_upper > hsv).all(): return 'Green'
    # else: return None

def avgColor(img, c, boxHeight, boxWidth):
    """TODO: Docstring for averageHSV.

    :img: HSV image from openCV
    :xy: x,y pixel location
    :returns: average HSV values near pixel

    """

    # print('Resolution: {}'.format(img.shape))

    # print(c)
    # print('Box h and w: {}'.format(boxHeight, boxWidth))

	# offsetW = int(boxWidth)/8
	# offsetH = int(boxHeight)/8

    offsetW = 100
    offsetH = 100

    minX = c[0] - offsetW
    maxX = c[0] + offsetW

    minY = c[1] - offsetH
    maxY = c[1] + offsetH

    h,w = img.shape[:2]

    if maxX > w:
		maxX  = w-1
    if maxY > h:
        maxY = h-1

    # print(minX, maxX)
    # print(minY, maxY)

    box = img[minY:maxY, minX:maxX]

    wMask = cv2.inRange(box, W_lower, W_upper)
    rMask = cv2.inRange(box, R_lower, R_upper)
    yMask = cv2.inRange(box, Y_lower, Y_upper)
    oMask = cv2.inRange(box, O_lower, O_upper)
    bMask = cv2.inRange(box, B_lower, B_upper)
    gMask = cv2.inRange(box, G_lower, G_upper)

    masks = [wMask, rMask, yMask, oMask, bMask, gMask]
    meanMasks = [np.mean(mask) for mask in masks]
    colors = ['white', 'red', 'yellow', 'orange', 'blue', 'green']
    cDict = {color: value for (color, value) in zip(colors, meanMasks)}

    #print('colorDict: {}'.format(cDict))

    color = max(cDict, key=cDict.get)


    # colorsDict = dict()
    # for i in range(minX, maxX):
    #     for j in range(minY, maxY):
    #         color = getColor(img[i,j])
    #         colorsDict[color] = 0

    # hsvs = []
    # for i in range(minX, maxX):
    #     for j in range(minY, maxY):
    #         hsvs.append(img[i,j])
    #         color = getColor(img[i,j])
    #         colorsDict[color] += 1

    # avgHsv = np.mean(hsvs, axis=0)
    # color = max(colorsDict, key=colorsDict.get)

    # h = np.mean(hues)
    # print('Colors Dict: {}'.format(colorsDict))
    # print('Centroid Hue: {}'.format(img[c]))
    # print('Average Hue: {}'.format(avgHsv))
    # print('Average Hue: {}'.format(h))

    return color


def colorCheck(img, centroids, cDict):
    plt.figure(figsize=(4,4))

    plt.imshow(img, 'gray')

    for c in centroids:
        plt.scatter(c[0], c[1], color=cDict[c])

    # x,y = zip(*centroids)
    # print(x,y)
    # plt.scatter(x,y)

    # _x,_y = zip(*bounds)
    # plt.scatter(_x,_y)

    plt.title('Centroids of Cube Stickers')
    plt.xticks([]), plt.yticks([])

    plt.show()


def main():

    kwargs = {

        'alpha': 2,         # contrast parameter
        'beta': 40,         # brightness parameter
        'gauss_ksize': 31,  # Gaussian blur kernel size (must be odd)
        'lap_ksize': 5      # Laplacian kernel size

    }


    face_colors = []

    img = cv2.imread('state_1/IMG_3045.JPG')
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    centroids, boxHeight, boxWidth = recognizeCube(img, **kwargs)


    colors = []
    centroidDict = {}
    for i,c in enumerate(centroids):
        # print('\nCube Face: {}'.format(i))
        # print('Centroid location: {}'.format(c))
        color = avgColor(hsvImg, c, boxHeight, boxWidth)
        centroidDict[c] = color
        colors.append(color)
        # pixel = hsvImg[c]
        # color = getColor(pixel)
        # print('color: {}'.format(color))
    print(colors)

    face_colors.append(colors)

    # print(face_colors)
    # plotCube(face_colors)
    colorCheck(rgbImg, centroids, centroidDict)

    # Pass face_colors to plotting function plotCube()



if __name__ == "__main__":
    main()

