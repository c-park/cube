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

def displayCentroids(img, centroids, bounds):
    plt.figure(figsize=(4,4))

    plt.imshow(img, 'gray')

    x,y = zip(*centroids)
    plt.scatter(x,y)

    _x,_y = zip(*bounds)
    plt.scatter(_x,_y)

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


class BoundingBox(object):
    """
    A 2D bounding box
    """
    def __init__(self, points):
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")
        self.minx, self.miny = float("inf"), float("inf")
        self.maxx, self.maxy = float("-inf"), float("-inf")
        for x, y in points:
            # Set min coords
            if x < self.minx:
                self.minx = x
            if y < self.miny:
                self.miny = y
            # Set max coords
            if x > self.maxx:
                self.maxx = x
            elif y > self.maxy:
                self.maxy = y
    @property
    def width(self):
        return self.maxx - self.minx
    @property
    def height(self):
        return self.maxy - self.miny
    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy)
