import numpy as np
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
import math

debug_ = True
show_ = True

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

def getImage(back_col):
    im_file = '\\data\\CubePictures\\' + back_col + ' Background\\IMG_2933.JPG'
    os.chdir('..')
    im_file = os.getcwd() + im_file
    if debug_:
        print 'Image File Location: ',im_file
    im_test = cv.imread(im_file)
    im_test = cv.resize(im_test, (0,0), fx=0.25, fy=0.25)
    if show_:
        cv.imshow('Test Image',im_test)
        cv.waitKey()
    p = im_test.shape
    return im_test


image = getImage('Black')
cv.imshow('TEST IMAGE',image)
cv.waitKey()
hsv_image = cv.cvtColor(image,cv.COLOR_BGR2HSV)
image = cv.blur(image,(5,5))
cv.imshow('BLUR IMAGE',image)
cv.waitKey()
cv.imshow('HSV IMAGE',hsv_image)
cv.waitKey()

s1, s2, v1, v2 = 0, 255, 220, 255
h1, h2 = 0, 179
hsv_lower = np.array([h1, s1, v1])
hsv_upper = np.array([h2, s2, v2])
white_mask = cv.inRange(hsv_image,hsv_lower,hsv_upper)
cv.imshow('WHITE MASK',white_mask)
cv.waitKey()
g = 10
mask_blur = cv.blur(white_mask,(g,g))
cv.imshow('MASK BLUR',mask_blur)
cv.waitKey()

corners = cv.cornerHarris(mask_blur,2,3,0.04)
print corners

edges = cv.Canny(mask_blur,100,200)
print edges
cv.imshow('EDGES',edges)
cv.waitKey()

fast = cv.FastFeatureDetector_create()
kp = fast.detect(mask_blur,None)
print kp[0].pt
img2 = cv.drawKeypoints(mask_blur,kp,None,color=(255,0,0))
cv.imshow('FEATURE',img2)
cv.waitKey()

y,x,c = img2.shape
print x,y,c

print img2[100,100]

key_points = [(0,0)]
for keypoint1 in kp:
    k1_xy = keypoint1.pt
    k1_list = []
    for keypoint2 in kp:
        k2_xy = keypoint2.pt
        if k1_xy != k2_xy:
            dist = math.sqrt( ((k1_xy[0]-k2_xy[0])**2)+((k1_xy[1]-k2_xy[1])**2) )
            #print dist
            if dist <= 10:
                k1_list.append(k2_xy)
        #print k1_list
        k1_len = len(k1_list) + 1
    x_sum = k1_xy[0]
    y_sum = k1_xy[1]
    for pt in k1_list:
        x_sum += pt[0]
        y_sum += pt[1]
    x_average = int(x_sum / k1_len)
    y_average = int(y_sum / k1_len)
    avg_point = (x_average,y_average)
    close = False
    for pt in key_points:
        dist2 = math.sqrt( ((avg_point[0]-pt[0])**2)+((avg_point[1]-pt[1])**2) )
        if dist2 <=30:
            close = True
    if close == False:
        key_points.append(avg_point)
key_points.remove((0,0))
print 'Len kp: ',len(kp)
print 'Len Keypoints: ',len(key_points)

for center in key_points:
    cv.circle(image,center,5,(255,0,0))
cv.imshow('IMAGE',image)
cv.waitKey()

x_min = 1000000
x_max = 0
y_min = 1000000
y_max = 0
for pt in key_points:
    pt_x = pt[0]
    pt_y = pt[1]
    if pt_x < x_min:
        x_min = pt_x
    if pt_x > x_max:
        x_max = pt_x
    if pt_y < y_min:
        y_min = pt_y
    if pt_y > y_max:
        y_max = pt_y
print x_min,x_max,y_min,y_max
x_minL = []
x_maxL = []
y_minL = []
y_maxL = []
t = 10
for pt in key_points:
    if pt[0] in range(x_min-t,x_min+t):
        print 'x_min: ',pt
        x_minL.append(pt)
        cv.circle(image,pt,5,(0,0,255),thickness=3)
    if pt[0] in range(x_max-t,x_max+t):
        print 'x_max: ',pt
        x_maxL.append(pt)
        cv.circle(image, pt, 5, (0, 0, 255), thickness=3)
    if pt[1] in range(y_min-t,y_min+t):
        print 'y_min: ',pt
        y_minL.append(pt)
        cv.circle(image, pt, 5, (0, 0, 255), thickness=3)
    if pt[1] in range(y_max-t,y_max+t):
        print 'y_max: ',pt
        y_maxL.append(pt)
        cv.circle(image, pt, 5, (0, 0, 255), thickness=3)

print 'X min: ',x_minL
print 'X max: ',x_maxL
print 'Y min: ',y_minL
print 'Y max: ',y_maxL
cv.imshow('IMAGE',image)
cv.waitKey()
TR = False
TL = False
BR = False
BL = False
corner_final = []
for pt in x_minL:
    if pt in y_maxL:
        corner_final.append(pt)
        TL = True
for pt in y_maxL:
    if pt in x_maxL:
        corner_final.append(pt)
        TR = True
for pt in x_maxL:
    if pt in y_minL:
        corner_final.append(pt)
        BR = True
for pt in y_minL:
    if pt in x_minL:
        corner_final.append(pt)
        BL = True
print corner_final
print TL,TR,BR,BL

if not TR:
    new_list = x_maxL + y_maxL
    TR_x = 0
    TR_y = 0
    for pt in new_list:
        pt_x = pt[0]
        pt_y = pt[1]
        if pt_x > TR_x:
            TR_x = pt_x
        if pt_y > TR_y:
            TR_y = pt_y
final_TR = (TR_x,TR_y)
corner_final.append(final_TR)

for final in corner_final:
    cv.circle(image, final, 5, (0, 255, 255), thickness=3)
cv.imshow('CORNER_FINAL',image)
cv.waitKey()

print corner_final

edge_list = []

dist_sum = 0
for pt in corner_final:
    for pt2 in corner_final:
        if pt != pt2:
            dist_sum += math.sqrt( ((pt[0]-pt2[0])**2)+((pt[1]-pt2[1])**2) )
dist_avg = dist_sum / (4 * (len(corner_final)-1))
print dist_avg
for pt in corner_final:
    for pt2 in corner_final:
        if (pt,pt2) not in edge_list and (pt2,pt) not in edge_list:
            if pt != pt2:
                dist_comp = math.sqrt( ((pt[0]-pt2[0])**2)+((pt[1]-pt2[1])**2) )
                print dist_comp
                if dist_comp < dist_avg:
                    edge_list.append((pt,pt2))
print edge_list

for edge in edge_list:
    cv.line(image,edge[0],edge[1],(0,255,255),thickness=3)
cv.imshow('LINES',image)
cv.waitKey()



'''
x_min = (10000,0)
x_max = (0,0)
y_min = (0,10000)
y_max = (0,0)
for pt in key_points:
    print pt
    pt_x = pt[0]
    pt_y = pt[1]
    if pt_x < x_min[0]:
        x_min = pt
    elif pt_x > x_max[0]:
        x_max = pt
    elif pt_y < y_min[1]:
        y_min = pt
    elif pt_y > y_max[1]:
        y_max = pt
corner_list = (x_min,x_max,y_min,y_max)
print corner_list
for corner in corner_list:
    cv.circle(image,corner,10,(0,255,0),thickness=5)
cv.imshow('IMAGE',image)
cv.waitKey()
'''

