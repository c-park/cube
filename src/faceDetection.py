#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: faceDetection.py
Author: Cade Parkison and Paul Wadsworth
Email: cadeparkison@gmail.com
Github: c-park
Description: Algorithm that recognizes a rubik's cube face and extracts the
             colors of each cubie. Initial algorithm based off Le Thanh Hoang paper.


Algorithm Overview:

    1.
    2.
    3.
    4.


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import sys

from utils import *

_DEBUG = True


def recognizeCube(img, **kwargs):
    """Given an image of a rubiks cube face, returns the x,y pixel coordinates
    of each cube sticker.

    :img: BGR image, read using openCV
    :returns: TODO

    """

    # convert to RGB
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert image to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adjust brightness and contrast
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    bcImg = cv2.addWeighted(grayImg, alpha, np.zeros(grayImg.shape, grayImg.dtype), 0, beta)

    # Gaussian blur
    g_ksize= kwargs['gauss_ksize']
    gaussImg = cv2.GaussianBlur(bcImg, (g_ksize,g_ksize), 0)

    # Laplacian operator
    lapImg = cv2.Laplacian(gaussImg,cv2.CV_16U, ksize=kwargs['lap_ksize'])

    # Dilation
    kernel = np.ones((30,30), np.uint8)
    dilateImg = cv2.dilate(lapImg, kernel, iterations=1)

    # Thresholding
    ret,thImg = cv2.threshold(dilateImg,40,255,cv2.THRESH_BINARY)

    # Opening
    kernel = np.ones((31,31),np.uint8)
    openImg = cv2.morphologyEx(thImg, cv2.MORPH_OPEN, kernel)

    # Second Dilation
    kernel = np.ones((80,80), np.uint8)
    dilateImg2 = cv2.dilate(openImg, kernel, iterations=1)

    # Find contours
    img2 = np.copy(dilateImg2)
    img2 = np.uint8(img2)
    contourImg, _contours, hierarchy = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size

    contours = np.array(_contours)
    filterContours = []

    for c in contours:
        c = np.array(c)
        c = c.reshape(c.shape[0], 2)
        filterContours.append(c)

    contours = []

    maxWidth = 450
    minWidth = 200
    maxHeight = 450
    minHeight = 200

    for c in filterContours:
        # print(len(c))
        # if _min <= len(c) <= _max:
        #     contours.append(c)
        box = BoundingBox(c)
        # print(box.width, box.height)
        if (minWidth <= box.width <= maxWidth) & (minHeight <= box.height <= maxHeight ):
            contours.append(c)


    # Find centroids of contours
    centroids = []
    for c in contours:
        centroid = centeroid(c)
        centroids.append(centroid)


    # average height and width of contours
    widths = []
    heights = []

    for c in contours:
        box = BoundingBox(c)
        widths.append(box.width)
        heights.append(box.height)

	boxWidth = int(np.mean(widths))
	boxHeight = int(np.mean(heights))
    # print(boxWidth, boxHeight)

    offsetW = 100
    offsetH = 100

    bounds = []
    for c in centroids:
        minX = c[0] - offsetW
        maxX = c[0] + offsetW

        minY = c[1] - offsetH
        maxY = c[1] + offsetH
        bounds.append([minX,minY])
        bounds.append([minX,maxY])
        bounds.append([maxX,minY])
        bounds.append([maxX,maxY])

    cList = sorted(centroids, key= lambda k: k[0])
    cList1 = cList[0:3]
    cList2 = cList[3:6]
    cList3 = cList[6:9]

    cList1_sorted = sorted(cList1, key =lambda k:k[1])
    cList2_sorted = sorted(cList2, key =lambda k:k[1])
    cList3_sorted = sorted(cList3, key =lambda k:k[1])

    # print(cList)
    # print(cList1)
    # print(cList2)
    # print(cList3)
    # print(cList1_sorted)
    # print(cList2_sorted)
    # print(cList3_sorted)

    sorted_centroids = cList1_sorted + cList2_sorted +cList3_sorted

	# print(boxWidth, boxHeight)

	# fig1 = plt.figure()
	# ax1 = fig1.add_subplot(111, aspect='equal')
	# ax1.imshow(rgbImg)
	# for c in filterContours:

		# ax1.add_patch(patches.Rectangle(c, int(boxWidth/4) ,int(boxHeight/4)  ))
	# plt.show()

    # print(avgWidth, avgHeight)

    # Display all steps
    if _DEBUG:
        pass
        # display(rgbImg, title='Original')
        # display(grayImg, title='Grayscale')
        # display(bcImg, title='Brightness and Contrast Adjusted')
        # display(gaussImg, title='Gaussian Blur')
        # display(lapImg, title='Laplacian Operator')
        # display(dilateImg, title='First Dilation')
        # display(thImg, title='Thresholded image')
        # display(dilateImg2, title='Second Dilation')
        # display(rgbImg, title='Contours', contours=_contours)
        # display(rgbImg, title='Filtered Contours', contours=contours)
        # displayCentroids(rgbImg, centroids, bounds)

    return sorted_centroids, boxHeight, boxWidth


def main():
    """TODO: Docstring for main.

    :args: TODO
    :returns: TODO

    """

    # Algorithm Parameters and thresholds
    kwargs = {

        'alpha': 2,         # contrast parameter
        'beta': 40,         # brightness parameter
        'gauss_ksize': 31,  # Gaussian blur kernel size (must be odd)
        'lap_ksize': 5      # Laplacian kernel size

    }



    # Read Test Images into memory

    # img1 = cv2.imread(test_path + 'IMG_2935.JPG')
    # img2 = cv2.imread(test_path + 'IMG_2958.JPG')
    # img3 = cv2.imread(test_path + 'IMG_2962.JPG')
    # img4 = cv2.imread(test_path + 'Mixed/IMG_2946.JPG')
    # testImages = [img1, img2, img3, img4]

    # img5 = cv2.imread('testImages/img5.jpg')
    # img6 = cv2.imread('testImages/img6.jpg')
    # img7 = cv2.imread('testImages/img7.jpg')
    # img8 = cv2.imread('testImages/img8.jpg')
    # testImages = [img5, img6, img7, img8]



    img = cv2.imread('state_1/IMG_3049.JPG')
    recognizeCube(img, **kwargs)



if __name__ == "__main__":
    # args = sys.argv[1:]
    main()

    # kwargs = {

    #     'alpha': 2,
    #     'beta': 40

    # }


    # test(**kwargs)
