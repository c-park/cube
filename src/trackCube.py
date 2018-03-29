"""
File: trackCube.py
Author: Paul Wadsworth
Email: wadspau2@gmail.com
Github: c-park
Description: Given an input image, locate a Rubik's cube

Usage:

"""

import numpy as np
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

debug_ = True
show_ = False
color_list = ['green','blue','red','white','yellow','orange']


green_up = np.array([102,255,255])
green_down = np.array([33,80,40])


class Frame:
    def __init__(self,video,frame,image,h,w,c):
        self.video = video
        self.frame = frame
        self.image = image
        self.image_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        self.image_hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
        self.image_small = cv.resize(self.image_gray,None,fx=0.5,fy=0.5)
        self.image_blur = cv.blur(self.image_gray,(1,1))
        self.height = h
        self.width = w
        self.channels = c

def getColor(color,alpha):
    h1,h2,s1,s2,v1,v2 = None,None,80,255,40,255
    if color == 'green':
        green = np.uint8([[[0,255,0]]])
        hsv_color = cv.cvtColor(green,cv.COLOR_BGR2HSV)
    if color == 'red':
        s1,s2,v1,v2 = 150,255,140,255
        h1,h2 = 160,179
        red = np.uint8([[[0,0,255]]])
        hsv_color = cv.cvtColor(red,cv.COLOR_BGR2HSV)
    if color == 'blue':
        s1,s2,v1,v2 = 80,255,80,255
        h1,h2 = 80,120
        blue = np.uint8([[[255,0,0]]])
        hsv_color = cv.cvtColor(blue,cv.COLOR_BGR2HSV)
    if color == 'white':
        s1,s2,v1,v2 = 30,50,0,200
        h1,h2 = 120,130
        white = np.uint8([[[255,255,255]]])
        hsv_color = cv.cvtColor(white,cv.COLOR_BGR2HSV)
    if color == 'yellow':
        s1,s2,v1,v2 = 100,255,140,255
        h1,h2 = 10,30
        yellow = np.uint8([[[255,255,0]]])
        hsv_color = cv.cvtColor(yellow,cv.COLOR_BGR2HSV)
    if color == 'orange':
        s1,s2,v1,v2 = 150,255,150,255
        h1,h2 = 0,10
        orange = np.uint8([[[255,165,0]]])
        hsv_color = cv.cvtColor(orange,cv.COLOR_BGR2HSV)
    if debug_:
        print 'Color: ',color
        print 'HSV Color Equivalent:',hsv_color[0][0][0]
        print 'HSV Saturation Equivalent: ',hsv_color[0][0][1]
        print 'HSV Value Equivalent: ',hsv_color[0][0][2]
    if h1 == None:
        hsv_lower = np.array([hsv_color[0][0][0] - alpha,s1,v1])
        hsv_upper = np.array([hsv_color[0][0][0] + alpha,s2,v2])
    else:
        hsv_lower = np.array([h1,s1,v1])
        hsv_upper = np.array([h2,s2,v2])
    return hsv_lower,hsv_upper

def getFrame(video,frame,show):
    im_file = '\\data\\video_' + str(video) + '\\frame' + str(frame) + '.jpg'
    os.chdir('..')
    im_file = os.getcwd() + im_file
    if debug_:
        print 'Image File Location: ',im_file
    im_test = cv.imread(im_file)
    if show:
        cv.imshow('Test Image',im_test)
        cv.waitKey()
    p = im_test.shape
    frame = Frame(video,frame,im_test,p[0],p[1],p[2])
    return frame

def getColorFigure(frame,im_list):
    scale = 2
    im_w = frame.width / scale
    im_h = frame.height / scale
    fig = plt.figure(figsize=(2,3))
    col = 3
    row = 3
    fig.add_subplot(row, col, 2)
    plt.axis('off')
    title = 'Original Frame'
    plt.title(title)
    plt.imshow(cv.cvtColor(frame.image, cv.COLOR_BGR2RGB))
    for i in range(4, col * row + 1):
        img = im_list[i - 4]
        fig.add_subplot(row, col, i)
        plt.axis('off')
        title = color_list[i - 4] + ' Mask'
        plt.title(title)
        plt.imshow(cv.cvtColor(img,cv.COLOR_GRAY2RGB))
    plt.show()

frame = getFrame(1,150,False)
if debug_:
    print 'Image Height: ',frame.height
    print 'Image Width: ',frame.width
    print 'Image # of Channels: ',frame.channels
    print 'Video: ',frame.video
    print 'Frame: ',frame.frame
    print 'Press any key to continue...'
if show_:
    cv.imshow('Debug Image',frame.image)
    cv.waitKey()
    cv.imshow('HSV Image',frame.image_hsv)
    cv.waitKey()
    cv.destroyAllWindows()

im_list = []
for color in color_list:
    hsv_lower,hsv_upper = getColor(color,20)
    if debug_:
        print 'HSV_lower: ',hsv_lower
        print 'HSV_upper: ',hsv_upper
    mask_image = cv.inRange(frame.image_hsv,hsv_lower,hsv_upper)
    im_list.append(mask_image)
    if show_:
        cv.imshow(('HSV ' + color + 'Mask'),mask_image)
        cv.waitKey()
        cv.destroyAllWindows()

if not show_:
    getColorFigure(frame,im_list)

imLine_list = []
for im in im_list:
    imLine_list.append(cv.Canny(im,0,10))
if show_:
    count = 0
    for im in imLine_list:
        print color_list[count]
        cv.imshow(('Edge ' + color_list[count] + 'Results'),imLine_list[count])
        cv.waitKey()
        cv.destroyAllWindows()
        count += 1

if not show_:
    getColorFigure(frame,imLine_list)


'''
cv.imshow('small',frame.image_small)
cv.waitKey()
cv.destroyAllWindows()

cv.imshow('blur',frame.image_blur)
cv.waitKey()
cv.destroyAllWindows()

laplacian = cv.Laplacian(frame.image_blur,cv.CV_64F)
cv.imshow('Laplacian', laplacian)
cv.waitKey()
cv.destroyAllWindows()
'''

'''
im = np.arange(6)
im.shape = 2,3
images = [im for i in range(20)]
fig = plt.figure(1,(4.,4.))
grid = ImageGrid(fig,111,nrows_ncols=(2,10),axes_pad=0)

for i in range(20):
    grid[i].imshow(images[i],cmap=plt.get_cmap('Greys_r'))
    grid[i].axis('off')
plt.show(block=True)
'''
'''
img_col = frame.image
green_list = []
green = (0,255,0)
for row in xrange(frame.height):
    for col in xrange(frame.width):
        if mask_image.item(row,col) == 255:
            green_list.append((row,col))
            img_col.itemset((row,col,0),green[0])
            img_col.itemset((row,col,1),green[1])
            img_col.itemset((row,col,2),green[2])
    

cv.imshow('GREEN TEST',img_col)
cv.waitKey()
cv.destroyAllWindows()
'''

'''
fast = cv.FastFeatureDetector_create()
kp = fast.detect(frame.image_gray)
im_out = np.zeros((frame.height,frame.width))
img2 = cv.drawKeypoints(frame.image_gray,kp,im_out,color=(255,0,0),)
cv.imshow('Test Image',img2)
cv.waitKey()
'''

'''

'''