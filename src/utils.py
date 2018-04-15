#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: utils.py
Author: Cade Parkison
Email: cadeparkison@gmail.com
Github: c-park
Description: Utility Functions
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

def centeroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def displayCentroids(img, centroids):
    plt.figure(figsize=(4,4))

    plt.imshow(img, 'gray')

    x,y = zip(*centroids)
    plt.scatter(x,y)

    plt.title('Centroids of Cube Stickers')
    plt.xticks([]), plt.yticks([])

    plt.show()

def display(img, title='', contours=None, cmap='gray'):
    """TODO: Docstring for displat.

    :img: TODO
    :returns: TODO

    """
    plt.figure(figsize=(4,4))

    if contours:
        rbgImg = np.copy(img)
        cont = cv2.drawContours(rbgImg, contours, -1, (255,0,0), 20)

        # cont_rgb = cv2.cvtColor(cont, cv2.COLOR_BGR2RGB)
        plt.imshow(cont, cmap=cmap)

    else:
        plt.imshow(img, cmap=cmap)

    plt.xticks([]), plt.yticks([])
    plt.title(title)

    plt.show()



def showImages(images, contourLists = None,title=''):
    plt.figure(figsize=(8,8))
    plt.suptitle(title)

    for j, img in enumerate(images):
        plt.subplot(2, 2, j+1)

        if contourLists:
            contour = contourLists[j]
            cont = cv2.drawContours(testImages2[j], contour, -1, (255,0,0), 20)
            plt.imshow(cont, cmap='gray')
        else:
            plt.imshow(img, cmap='gray')

        plt.xticks([]), plt.yticks([])
        plt.title('Image {}'.format(j+1))

    plt.show()


def showContours(images, contourLists = None,title=''):
    plt.figure(figsize=(12,12))
    plt.suptitle(title)

    for j, img in enumerate(images):
        plt.subplot(2, 2, j+1)
        contour = contourLists[j]
        cont = cv2.drawContours(testImages2[j], contour, -1, (255,0,0), 20)
        cont_rgb = cv2.cvtColor(cont, cv2.COLOR_BGR2RGB)
        plt.imshow(cont_rgb, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title('Image {}'.format(j+1))

    plt.show()
