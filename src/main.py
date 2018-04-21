#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: main.py
Author: Cade Parkison
Email: cadeparkison@gmail.com
Github: c-park
Description: Main file to run Rubik's cube algorithm
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from faceDetection import recognizeCube
from colorExtraction import getColor, avgColor
from plotting import plotCube, plotCubes
from utils import *


def main():
    """TODO: docstring """

    # Algorithm Parameters and thresholds
    kwargs = {

        'alpha': 2,         # contrast parameter
        'beta': 40,         # brightness parameter
        'gauss_ksize': 31,  # Gaussian blur kernel size (must be odd)
        'lap_ksize': 5      # Laplacian kernel size

    }

    # Read Test Images into memory

    img1 = cv2.imread('testImages/img2.jpg')
    # img2 = cv2.imread(test_path + 'IMG_2958.JPG')
    # img3 = cv2.imread(test_path + 'IMG_2962.JPG')
    # img4 = cv2.imread(test_path + 'Mixed/IMG_2946.JPG')
    # testImages = [img1, img2, img3, img4]

    img5 = cv2.imread('testImages/img5.jpg')
    img6 = cv2.imread('testImages/img6.jpg')
    img7 = cv2.imread('testImages/img7.jpg')
    img8 = cv2.imread('testImages/img8.jpg')
    testImages = [img5, img6, img7, img8]


    # for img in testImages:

    centroids, boxHeight, boxWidth = recognizeCube(img1, **kwargs)

    hsvImg = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    for i,c in enumerate(centroids):
        # print('\nCube Face: {}'.format(i))
        # print('Centroid location: {}'.format(c))
        color = avgColor(hsvImg, c, boxHeight, boxWidth)
        # pixel = hsvImg[c]
        # color = getColor(pixel)
        # print('color: {}'.format(color))


def cube():

    kwargs = {

        'alpha': 2,         # contrast parameter
        'beta': 40,         # brightness parameter
        'gauss_ksize': 31,  # Gaussian blur kernel size (must be odd)
        'lap_ksize': 5      # Laplacian kernel size

    }

    s1_dir = 'state_1/'
    s2_dir = 'state_2/'

    s1_Imgs = []
    s1_hsvImgs = []

    s2_Imgs = []
    s2_hsvImgs = []


    face_colors = []

    for fname in os.listdir(s1_dir):
        # print(s1_dir + fname)
        fdir = s1_dir + fname
        print('\nProcessing file: {}'.format(fdir))

        img = cv2.imread(fdir)
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        centroids, boxHeight, boxWidth = recognizeCube(img, **kwargs)


        colors = []
        for i,c in enumerate(centroids):
            # print('\nCube Face: {}'.format(i))
            # print('Centroid location: {}'.format(c))
            color = avgColor(hsvImg, c, boxHeight, boxWidth)
            colors.append(color)
            # pixel = hsvImg[c]
            # color = getColor(pixel)
            # print('color: {}'.format(color))
        print(colors)

        face_colors.append(colors)

        name = os.path.splitext(fname)[0]
        title = 'results_' + name
        # plotCube(rgbImg, colors, title)

    plotCubes(face_colors, 'state_1_all_faces_v1')

    # Pass face_colors to plotting function plotCube()


    # for fname in os.listdir(s2_dir):


if __name__ == "__main__":
    # main()
    cube()
