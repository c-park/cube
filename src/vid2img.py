#!/usr/bin/env python

"""
File: vid2img.py
Author: Cade Parkison
Email: cadeparkison@gmail.com
Github: c-park
Description: Given an imput video, create .jpg images of each frame and add to subfolder

Usage:

    - python vid2img.py <image> <folder name>
"""

import numpy as np
import cv2 as cv
import sys
import os


def video2images(f):
	cap = cv.VideoCapture(f+'/' + f + '.webm')
	success,image = cap.read()
	count = 0
	success = True

	while success:
		cv.imwrite(f + '/frame%d.jpg' % count, image)
		success,image = cap.read()
		count += 1

if __name__ == "__main__":
	f = sys.argv[1]
	video2images(f)

