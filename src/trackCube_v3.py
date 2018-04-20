import numpy as np
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
import math

debug_ = True
show_ = False
init_ = False
g_ = 10 # Gaussian Blur Level
NNT_ = 10 # Nearest neighbor threshold
KT_ = 30 # Nearest keypoint threshold
MMT_ = 10 # Min/Max Threshold

color_list = ['White','Red','Yellow','Orange','Blue','Green']
W_lower,W_upper = np.array([0,0,196]), np.array([179,21,255])
R_lower,R_upper = np.array([0,203,204]), np.array([179,222,255])
Y_lower,Y_upper = np.array([23,137,199]), np.array([39,198,255])
O_lower,O_upper = np.array([7,188,188]), np.array([37,255,255])
B_lower,B_upper = np.array([94,229,167]), np.array([127,255,213])
G_lower,G_upper = np.array([62,29,131]), np.array([73,255,210])


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

class Face:
    def __init__(self,color):
        self.colors = np.array([[color,color,color],[color,color,color],[color,color,color]],dtype=object)
        self.rotation = 0 # Can be 0, 90, 180, or 270
        self.corners = []
        self.edges = []
        self.thirds = []
        self.grid = []
        self.areas = []
        self.plot = []

def getFrame(video,frame):
    im_file = '\\data\\video_' + str(video) + '\\frame' + str(frame) + '.jpg'
    os.chdir('..')
    im_file = os.getcwd() + im_file
    if debug_:
        print 'Image File Location: ',im_file
    im_test = cv.imread(im_file)
    if show_:
        cv.imshow('Test Image',im_test)
        cv.waitKey()
    p = im_test.shape
    frame = Frame(video,frame,im_test,p[0],p[1],p[2])
    return frame

def getImage(back_col,im_num,solid):
    if solid:
        im_file = '\\data\\CubePictures\\' + back_col + ' Background\\IMG_' + str(im_num) + '.JPG'
    if not solid:
        im_file = '\\data\\CubePictures\\' + back_col + ' Background\\Mixed\\IMG_' + str(im_num) + '.JPG'
    os.chdir('..')
    im_file = os.getcwd() + im_file
    if debug_:
        print 'Image File Location: ',im_file
    im_test = cv.imread(im_file)
    im_test = cv.resize(im_test, (0,0), fx=0.25, fy=0.25)
    if show_:
        cv.imshow('getImage',im_test)
        cv.waitKey()
    p = im_test.shape
    image = Frame(None,None,im_test,p[0],p[1],p[2])
    return image

def getMask(im_hsv):
    W_mask = cv.inRange(im_hsv,W_lower,W_upper)
    R_mask = cv.inRange(im_hsv,R_lower,R_upper)
    Y_mask = cv.inRange(im_hsv,Y_lower,Y_upper)
    O_mask = cv.inRange(im_hsv,O_lower,O_upper)
    B_mask = cv.inRange(im_hsv,B_lower,B_upper)
    G_mask = cv.inRange(im_hsv,G_lower,G_upper)
    if show_:
        cv.imshow('White Mask',W_mask)
        cv.imshow('Red Mask',R_mask)
        cv.imshow('Yellow Mask',Y_mask)
        cv.imshow('Orange Mask',O_mask)
        cv.imshow('Blue Mask',B_mask)
        cv.imshow('Green Mask',G_mask)
        cv.waitKey()
    mask = W_mask + R_mask + Y_mask + O_mask + B_mask + G_mask
    return [W_mask,R_mask,Y_mask,O_mask,B_mask,G_mask],mask

def showMaskFigure(frame,im_list):
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

def nothing(x):
    pass

def getThresholds(image):
    W_lower, W_upper = np.array([0, 0, 0]), np.array([0, 0, 0])
    R_lower, R_upper = np.array([0, 0, 0]), np.array([0, 0, 0])
    Y_lower, Y_upper = np.array([0, 0, 0]), np.array([0, 0, 0])
    O_lower, O_upper = np.array([0, 0, 0]), np.array([0, 0, 0])
    B_lower, B_upper = np.array([0, 0, 0]), np.array([0, 0, 0])
    G_lower, G_upper = np.array([0, 0, 0]), np.array([0, 0, 0])
    cv.namedWindow('HSV Threshold Selection')
    cv.namedWindow('HSV Assignment')
    hh = 'Upper H'
    hl = 'Lower H'
    sh = 'Upper S'
    sl = 'Lower S'
    vh = 'Upper V'
    vl = 'Lower V'
    W = 'White'
    R = 'Red'
    Y = 'Yellow'
    O = 'Orange'
    B = 'Blue'
    G = 'Green'

    cv.createTrackbar(hl, 'HSV Threshold Selection', 0, 179, nothing)
    cv.createTrackbar(hh, 'HSV Threshold Selection', 179, 179, nothing)
    cv.createTrackbar(sl, 'HSV Threshold Selection', 0, 255, nothing)
    cv.createTrackbar(sh, 'HSV Threshold Selection', 255, 255, nothing)
    cv.createTrackbar(vl, 'HSV Threshold Selection', 0, 255, nothing)
    cv.createTrackbar(vh, 'HSV Threshold Selection', 255, 255, nothing)
    cv.createTrackbar(W, 'HSV Assignment', 0, 1, nothing)
    cv.createTrackbar(R, 'HSV Assignment', 0, 1, nothing)
    cv.createTrackbar(Y, 'HSV Assignment', 0, 1, nothing)
    cv.createTrackbar(O, 'HSV Assignment', 0, 1, nothing)
    cv.createTrackbar(B, 'HSV Assignment', 0, 1, nothing)
    cv.createTrackbar(G, 'HSV Assignment', 0, 1, nothing)


    while(1):
        cv.imshow('Original Image',image.image)
        #cv.imshow('HSV Image',image.image_hsv)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        #read trackbar positions for all
        hul=cv.getTrackbarPos(hl, 'HSV Threshold Selection')
        huh=cv.getTrackbarPos(hh, 'HSV Threshold Selection')
        sal=cv.getTrackbarPos(sl, 'HSV Threshold Selection')
        sah=cv.getTrackbarPos(sh, 'HSV Threshold Selection')
        val=cv.getTrackbarPos(vl, 'HSV Threshold Selection')
        vah=cv.getTrackbarPos(vh, 'HSV Threshold Selection')
        Wv = cv.getTrackbarPos(W, 'HSV Assignment')
        Rv = cv.getTrackbarPos(R, 'HSV Assignment')
        Yv = cv.getTrackbarPos(Y, 'HSV Assignment')
        Ov = cv.getTrackbarPos(O, 'HSV Assignment')
        Bv = cv.getTrackbarPos(B, 'HSV Assignment')
        Gv = cv.getTrackbarPos(G, 'HSV Assignment')

        if Wv == 1:
            W_lower = np.array([hul, sal, val])
            W_upper = np.array([huh, sah, vah])
        if Rv == 1:
            R_lower = np.array([hul, sal, val])
            R_upper = np.array([huh, sah, vah])
        if Yv == 1:
            Y_lower = np.array([hul, sal, val])
            Y_upper = np.array([huh, sah, vah])
        if Ov == 1:
            O_lower = np.array([hul, sal, val])
            O_upper = np.array([huh, sah, vah])
        if Bv == 1:
            B_lower = np.array([hul, sal, val])
            B_upper = np.array([huh, sah, vah])
        if Gv == 1:
            G_lower = np.array([hul, sal, val])
            G_upper = np.array([huh, sah, vah])



        #make array for final values
        HSVLOW=np.array([hul,sal,val])
        HSVHIGH=np.array([huh,sah,vah])

        #apply the range on a mask
        mask = cv.inRange(image.image_hsv,HSVLOW, HSVHIGH)
        res = cv.bitwise_and(image.image_hsv,image.image_hsv, mask =mask)
        cv.imshow('Current Mask',res)
    cv.destroyAllWindows()
    if debug_:
        print 'W_lower: ',W_lower
        print 'W_upper: ',W_upper
        print 'R_lower: ',R_lower
        print 'R_upper: ',R_upper
        print 'Y_lower: ',Y_lower
        print 'Y_upper: ',Y_upper
        print 'O_lower: ',O_lower
        print 'O_upper: ',O_upper
        print 'B_lower: ',B_lower
        print 'B_upper: ',B_upper
        print 'G_lower: ',G_lower
        print 'G_upper: ',G_upper

def getFeatures(mask_blur,im):
    fast = cv.FastFeatureDetector_create()
    kp = fast.detect(mask_blur, None) # List of keypoints from FAST algorithm
    im_features = cv.drawKeypoints(im,kp,None,color=(255,0,0))
    if show_:
        cv.imshow('FEATURES', im_features)
        cv.waitKey()

    key_points = [(0, 0)] # Initiate keypoint list with an initial 'guinea pig' key point
    for keypoint1 in kp:
        k1_xy = keypoint1.pt # Gives the (x,y) coordinate of the keypoint
        k1_list = [] # Keypoint nearest neighbor list

        for keypoint2 in kp: # Compare the first keypoint to all other keypoints
            k2_xy = keypoint2.pt # Gives the (x,y) coordinate of the comparing keypoint
            if k1_xy != k2_xy: # If they are not the same
                dist = math.sqrt(((k1_xy[0] - k2_xy[0]) ** 2) + ((k1_xy[1] - k2_xy[1]) ** 2)) # Computer the distance between keypoints
                if dist <= NNT_: # If within threshold, add it to the nearest neighbor list
                    k1_list.append(k2_xy)
            k1_len = len(k1_list) + 1

        x_sum = k1_xy[0] # Initialize x_sum for averaging
        y_sum = k1_xy[1] # Initialize y_sum for averaging
        for pt in k1_list: # Sum all x and y values for all points in the keypoint nearest neighbor list
            x_sum += pt[0]
            y_sum += pt[1]
        x_average = int(x_sum / k1_len) # Calculate the x average of points
        y_average = int(y_sum / k1_len) # Calculate the y average of points
        avg_point = (x_average, y_average)
        close = False
        for pt in key_points: # Calcualte the distance from the new average point to each previously cleaned keypoint
            dist2 = math.sqrt(((avg_point[0] - pt[0]) ** 2) + ((avg_point[1] - pt[1]) ** 2))
            if dist2 <= KT_:
                close = True
        if close == False:
            key_points.append(avg_point)

    key_points.remove((0, 0)) # Remove the initialization point from keypoint list
    if debug_:
        print 'Len kp: ', len(kp)
        print 'Len Keypoints: ', len(key_points)
    for center in key_points:
        cv.circle(im, center, 5, (255, 0, 0), thickness = 2)
    if show_:
        cv.imshow('CLEANED KEYPOINTS', im)
        cv.waitKey()

    x_min = 1000000
    x_max = 0
    y_min = 1000000
    y_max = 0
    for pt in key_points: # Fine the min and max of both x and y coordinates for all points
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
    if debug_:
        print 'X_min: ',x_min
        print 'X_max: ',x_max
        print 'Y_min: ',y_min
        print 'Y_max: ',y_max
    x_minL,x_maxL,y_minL,y_maxL = [],[],[],[]
    t = MMT_ # Min/Max threshold
    for pt in key_points: # Create lists of all points that fall within thresholds for x/y min/max
        if pt[0] in range(x_min - t, x_min + t):
            if debug_:
                print 'x_min: ', pt
            x_minL.append(pt)
            cv.circle(im, pt, 5, (0, 0, 255), thickness=3)
        if pt[0] in range(x_max - t, x_max + t):
            if debug_:
                print 'x_max: ', pt
            x_maxL.append(pt)
            cv.circle(im, pt, 5, (0, 0, 255), thickness=3)
        if pt[1] in range(y_min - t, y_min + t):
            if debug_:
                print 'y_min: ', pt
            y_minL.append(pt)
            cv.circle(im, pt, 5, (0, 0, 255), thickness=3)
        if pt[1] in range(y_max - t, y_max + t):
            if debug_:
                print 'y_max: ', pt
            y_maxL.append(pt)
            cv.circle(im, pt, 5, (0, 0, 255), thickness=3)

    if debug_:
        print 'X min: ', x_minL
        print 'X max: ', x_maxL
        print 'Y min: ', y_minL
        print 'Y max: ', y_maxL
    if show_:
        cv.imshow('X/Y Min/Max', im)
        cv.waitKey()

    TR,TL,BR,BL = False,False,False,False
    corner_final = []
    if len(x_minL) > 1:
        for pt in x_minL:
            if pt in y_maxL:
                corner_final.append(pt)
                TL = True
    else:
        corner_final.append(x_minL[0])

    if len(y_maxL) > 1:
        for pt in y_maxL:
            if pt in x_maxL:
                corner_final.append(pt)
                TR = True
    else:
        corner_final.append(y_maxL[0])

    if len(x_maxL) > 1:
        for pt in x_maxL:
            if pt in y_minL:
                corner_final.append(pt)
                BR = True
    else:
        corner_final.append(x_maxL[0])

    if len(y_minL) > 1:
        for pt in y_minL:
            if pt in x_minL:
                corner_final.append(pt)
                BL = True
    else:
        corner_final.append(y_minL[0])

    if debug_:
        print 'Final Corners: ',corner_final
        print 'Corner Status: ',TL, TR, BR, BL

    if (TR == False) & (len(y_maxL) > 1):
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
        final_TR = (TR_x, TR_y)
        corner_final.append(final_TR)
    if (TL == False) & (len(x_minL) > 1):
        new_list = x_minL + y_maxL
        TL_x = 100000
        TL_y = 0
        for pt in new_list:
            pt_x = pt[0]
            pt_y = pt[1]
            if pt_x < TL_x:
                TL_x = pt_x
            if pt_y > TL_y:
                TL_y = pt_y
        final_TL = (TL_x,TL_y)
        corner_final.append(final_TL)
    if (BR == False) & (len(x_maxL) > 1):
        new_list = x_maxL + y_minL
        TL_x = 0
        TL_y = 10000000
        for pt in new_list:
            pt_x = pt[0]
            pt_y = pt[1]
            if pt_x > TL_x:
                TL_x = pt_x
            if pt_y < TL_y:
                TL_y = pt_y
        final_TL = (TL_x,TL_y)
        corner_final.append(final_TL)
    if (BL == False) & (len(y_minL) > 1):
        new_list = x_minL + y_minL
        TL_x = 100000
        TL_y = 100000
        for pt in new_list:
            pt_x = pt[0]
            pt_y = pt[1]
            if pt_x < TL_x:
                TL_x = pt_x
            if pt_y < TL_y:
                TL_y = pt_y
        final_TL = (TL_x,TL_y)
        corner_final.append(final_TL)


    if debug_:
        print 'Final Corners: ',corner_final
        print 'Num of Corners: ',len(corner_final)

    for final in corner_final:
        cv.circle(im, final, 5, (0, 255, 255), thickness=3)
    if show_:
        cv.imshow('CORNER_FINAL', im)
        cv.waitKey()

    edge_list = []
    dist_sum = 0
    for pt in corner_final:
        for pt2 in corner_final:
            if pt != pt2:
                dist_sum += math.sqrt(((pt[0] - pt2[0]) ** 2) + ((pt[1] - pt2[1]) ** 2))
    dist_avg = dist_sum / (4 * (len(corner_final) - 1))
    if debug_:
        print 'Avg Distance: ',dist_avg
    for pt in corner_final:
        for pt2 in corner_final:
            if (pt, pt2) not in edge_list and (pt2, pt) not in edge_list:
                if pt != pt2:
                    dist_comp = math.sqrt(((pt[0] - pt2[0]) ** 2) + ((pt[1] - pt2[1]) ** 2))
                    #print dist_comp
                    if dist_comp < dist_avg:
                        edge_list.append((pt, pt2))
    if debug_:
        print 'Edge List: ',edge_list
    for edge in edge_list:
        cv.line(im, edge[0], edge[1], (0, 255, 255), thickness=3)
    if show_:
        cv.imshow('EDGES', im)
        cv.waitKey()

    return edge_list,im,dist_avg,corner_final

def getSubplot(edges,im,dist_avg):
    thirds = []
    centers = []
    for edge in edges:
        if debug_:
            print 'Edge: ',edge
        pt1,pt2 = edge[0],edge[1]
        pts = [pt1,pt2]
        pts_x = [pt1[0],pt2[0]]
        pts_y = [pt1[1],pt2[1]]
        e_len = (abs(pt1[0] - pt2[0]),abs(pt1[1] - pt2[1]))
        if debug_:
            print 'E_len: ',e_len
        third1,third2 = ((e_len[0] / 3),(e_len[1] / 3)),((2 * e_len[0] / 3),(2 * e_len[1] / 3))
        center1,center2,center3 = ((e_len[0] / 6),(e_len[1] / 6)),((e_len[0] / 2),(e_len[1] / 2)),((5 * e_len[0] / 6),(5 * e_len[1] / 6))
        if debug_:
            print 'Thirds: ',third1,third2
        x_data = (min(pts_x),np.argmin(pts_x))
        y_data = (min(pts_y),np.argmin(pts_y))
        pt1_add = (x_data[0],y_data[0])
        if x_data[1] == y_data[1]:
            third1 = ((third1[0] + pt1_add[0]), (third1[1] + pt1_add[1]))
            third2 = ((third2[0] + pt1_add[0]), (third2[1] + pt1_add[1]))
            center1 = ((center1[0] + pt1_add[0]), (center1[1] + pt1_add[1]))
            center2 = ((center2[0] + pt1_add[0]), (center2[1] + pt1_add[1]))
            center3 = ((center3[0] + pt1_add[0]), (center3[1] + pt1_add[1]))
        else:
            if y_data[1] == 0:
                pt1_add = (x_data[0],pts_y[1])
            else:
                pt1_add = (x_data[0],pts_y[0])
            third1 = ((third1[0] + pt1_add[0]), (-third1[1] + pt1_add[1]))
            third2 = ((third2[0] + pt1_add[0]), (-third2[1] + pt1_add[1]))
            center1 = ((center1[0] + pt1_add[0]), (-center1[1] + pt1_add[1]))
            center2 = ((center2[0] + pt1_add[0]), (-center2[1] + pt1_add[1]))
            center3 = ((center3[0] + pt1_add[0]), (-center3[1] + pt1_add[1]))
        if debug_:
            print 'Pt_add: ',pt1_add
        thirds.append(third1)
        thirds.append(third2)
        centers.append(center1)
        centers.append(center2)
        centers.append(center3)
    if debug_:
        print 'Third List: ',thirds
    for third in thirds:
        cv.circle(im, third, 10, (0, 255, 0), thickness=3)
    for center in centers:
        cv.circle(im, center, 10, (255, 255, 0), thickness=3)
    if show_:
        cv.imshow('THIRDS', im)
        cv.waitKey()
    subplot = []
    for pt in thirds:
        for pt2 in thirds:
            if pt != pt2:
                dist_comp = math.sqrt(((pt[0] - pt2[0]) ** 2) + ((pt[1] - pt2[1]) ** 2))
                t1 = -50
                t2 = 75
                if dist_comp < (dist_avg + t1):
                    if dist_comp > (dist_avg - t2):
                        #print dist_avg,dist_comp
                        subplot.append((pt, pt2))
    subplot_centers = []
    for pt in centers:
        for pt2 in centers:
            if pt != pt2:
                dist_comp = math.sqrt(((pt[0] - pt2[0]) ** 2) + ((pt[1] - pt2[1]) ** 2))
                t1 = -50
                t2 = 63
                if dist_comp < (dist_avg + t1):
                    if dist_comp > (dist_avg - t2):
                        #print dist_avg,dist_comp
                        subplot_centers.append((pt, pt2))
    if debug_:
        print 'Subplot: ',subplot
    for edge in subplot:
        cv.line(im, edge[0], edge[1], (0, 255, 0), thickness=3)
    for edge in subplot_centers:
        cv.line(im, edge[0], edge[1], (255, 255, 0), thickness=3)
    if show_:
        cv.imshow('SUBPLOT',im)
        cv.waitKey()
    subplot_final = []
    for line in subplot:
        for line2 in subplot:
            if line == (line2[1],line2[0]):
                if line not in subplot_final:
                    if (line[1],line[0]) not in subplot_final:
                        subplot_final.append(line)

    return subplot_final,im

def getCenters(corners):
    corners = sorted(corners,key=lambda k: [k[1]])
    if debug_:
        print 'Sorted Corners: ',corners
    pt1,pt2 = corners[0],corners[1]
    deltaY = float(abs(pt2[1] - pt1[1]))
    deltaX = float(abs(pt2[0] - pt1[0]))
    theta = math.atan((deltaY / deltaX))
    edge_len = math.sqrt((deltaY * deltaY) + (deltaX * deltaX))
    if debug_:
        print 'Delta Y: ',deltaY
        print 'Delta X: ',deltaX
        print 'Theta: ',theta
    if pt1[0] < pt2[0]:
        c1 = ((deltaX / 6) + pt1[0],(deltaY / 6) + pt1[1])
        c2 = ((deltaX / 2) + pt1[0],(deltaY / 2) + pt1[1])
        c3 = ((5 * deltaX / 6) + pt1[0],(5 * deltaY / 6) + pt1[1])
        p1,p2,p3 = (int(c1[0]),int(c1[1])),(int(c2[0]),int(c2[1])),(int(c3[0]),int(c3[1]))
        c1 = (-(deltaY / 6) + p1[0],(deltaX / 6) + p1[1])
        c2 = (-(deltaY / 2) + p1[0],(deltaX / 2) + p1[1])
        c3 = (-(5 * deltaY / 6) + p1[0],(5 * deltaX / 6) + p1[1])
        c4 = (-(deltaY / 6) + p2[0],(deltaX / 6) + p2[1])
        c5 = (-(deltaY / 2) + p2[0],(deltaX / 2) + p2[1])
        c6 = (-(5 * deltaY / 6) + p2[0],(5 * deltaX / 6) + p2[1])
        c7 = (-(deltaY / 6) + p3[0],(deltaX / 6) + p3[1])
        c8 = (-(deltaY / 2) + p3[0],(deltaX / 2) + p3[1])
        c9 = (-(5 * deltaY / 6) + p3[0],(5 * deltaX / 6) + p3[1])
        p1,p2,p3 = (int(c1[0]),int(c1[1])),(int(c2[0]),int(c2[1])),(int(c3[0]),int(c3[1]))
        p4,p5,p6 = (int(c4[0]),int(c4[1])),(int(c5[0]),int(c5[1])),(int(c6[0]),int(c6[1]))
        p7,p8,p9 = (int(c7[0]),int(c7[1])),(int(c8[0]),int(c8[1])),(int(c9[0]),int(c9[1]))
    else:
        c1 = (-(deltaX / 6) + pt1[0],(deltaY / 6) + pt1[1])
        c2 = (-(deltaX / 2) + pt1[0],(deltaY / 2) + pt1[1])
        c3 = ((-5 * deltaX / 6) + pt1[0],(5 * deltaY / 6) + pt1[1])
        p1,p2,p3 = (int(c1[0]),int(c1[1])),(int(c2[0]),int(c2[1])),(int(c3[0]),int(c3[1]))
        c1 = ((deltaY / 6) + p1[0],(deltaX / 6) + p1[1])
        c2 = ((deltaY / 2) + p1[0],(deltaX / 2) + p1[1])
        c3 = ((5 * deltaY / 6) + p1[0],(5 * deltaX / 6) + p1[1])
        c4 = ((deltaY / 6) + p2[0],(deltaX / 6) + p2[1])
        c5 = ((deltaY / 2) + p2[0],(deltaX / 2) + p2[1])
        c6 = ((5 * deltaY / 6) + p2[0],(5 * deltaX / 6) + p2[1])
        c7 = ((deltaY / 6) + p3[0],(deltaX / 6) + p3[1])
        c8 = ((deltaY / 2) + p3[0],(deltaX / 2) + p3[1])
        c9 = ((5 * deltaY / 6) + p3[0],(5 * deltaX / 6) + p3[1])
        p1,p2,p3 = (int(c1[0]),int(c1[1])),(int(c2[0]),int(c2[1])),(int(c3[0]),int(c3[1]))
        p4,p5,p6 = (int(c4[0]),int(c4[1])),(int(c5[0]),int(c5[1])),(int(c6[0]),int(c6[1]))
        p7,p8,p9 = (int(c7[0]),int(c7[1])),(int(c8[0]),int(c8[1])),(int(c9[0]),int(c9[1]))
    center_list = [p1,p2,p3,p4,p5,p6,p7,p8,p9]
    for pt in center_list:
        cv.circle(im, pt, 10, (255, 255, 0), thickness=3)
    if show_:
        cv.imshow('PTS', im)
        cv.waitKey()
    return [p1,p2,p3,p4,p5,p6,p7,p8,p9]

def getColor(pt,image_hsv):
    color = None
    pt = (pt[0],pt[1])
    h,s,v = image_hsv[pt]
    cv.circle(image_hsv,pt,5,(255,0,0),thickness=3)
    cv.imshow('TEST',image_hsv)
    cv.waitKey()
    if h in xrange(W_lower[0],W_upper[0]):
        print 'h'
        if s in xrange(W_lower[1],W_upper[1]):
            print 's'
            if v in xrange(W_lower[2],W_upper[2]):
                print 'v'
                color = 'white'
    if h in xrange(R_lower[0],R_upper[0]):
        print 'h'
        if s in xrange(R_lower[1],R_upper[1]):
            print 's'
            if v in xrange(R_lower[2],R_upper[2]):
                print 'v'
                color = 'red'
    if h in xrange(Y_lower[0],Y_upper[0]):
        print 'h'
        if s in xrange(Y_lower[1],Y_upper[1]):
            print 's'
            if v in xrange(Y_lower[2],Y_upper[2]):
                print 'v'
                color = 'yellow'
    if h in xrange(O_lower[0],O_upper[0]):
        print 'h'
        if s in xrange(O_lower[1],O_upper[1]):
            print 's'
            if v in xrange(O_lower[2],O_upper[2]):
                print 'v'
                color = 'orange'
    if h in xrange(B_lower[0],B_upper[0]):
        print 'h'
        if s in xrange(B_lower[1],B_upper[1]):
            print 's'
            if v in xrange(B_lower[2],B_upper[2]):
                print 'v'
                color = 'blue'
    if h in xrange(G_lower[0],G_upper[0]):
        print 'h'
        if s in xrange(G_lower[1],G_upper[1]):
            print 's'
            if v in xrange(G_lower[2],G_upper[2]):
                print 'v'
                color = 'green'

    print h,s,v
    print color
    return color

def getMaskColor(pt,im_hsv):
    W_mask = cv.inRange(im_hsv,W_lower,W_upper)
    R_mask = cv.inRange(im_hsv,R_lower,R_upper)
    Y_mask = cv.inRange(im_hsv,Y_lower,Y_upper)
    O_mask = cv.inRange(im_hsv,O_lower,O_upper)
    B_mask = cv.inRange(im_hsv,B_lower,B_upper)
    G_mask = cv.inRange(im_hsv,G_lower,G_upper)
    W_list = []
    R_list = []
    Y_list = []
    O_list = []
    B_list = []
    G_list = []
    r = 5
    for x in xrange(-r,r):
        for y in xrange(-r,r):
            pt = (pt[0]+x,pt[1]+y)
            if W_mask[pt] == 255:
                W_list.append('white')
            elif R_mask[pt] == 255:
                R_list.append('red')
            elif Y_mask[pt] == 255:
                Y_list.append('yellow')
            elif O_mask[pt] == 255:
                O_list.append('orange')
            elif B_mask[pt] == 255:
                B_list.append('blue')
            elif G_mask[pt] == 255:
                G_list.append('green')
    W,R,Y,O,B,G = len(W_list),len(R_list),len(Y_list),len(O_list),len(B_list),len(G_list)
    print W,R,Y,O,B,G
    res = np.argmax((W,R,Y,O,B,G))
    if res == 0:
        return 'White'
    if res == 1:
        return 'Red'
    if res == 2:
        return 'Yellow'
    if res == 3:
        return 'Orange'
    if res == 4:
        return 'Blue'
    if res == 5:
        return 'Green'

image = getImage('Black',2948,False)
#image = getImage('Black',2958,True)
#frame = getFrame(1,100)
if init_:
    getThresholds(image)
mask_list,mask = getMask(image.image_hsv)
showMaskFigure(image,mask_list)
mask_blur = cv.blur(mask,(g_,g_))
edges,im,dist_avg,corners = getFeatures(mask_blur,image.image)
pts = getCenters(corners)
center_pt = pts[1]
color_list1 = []
for pt in pts:
    color = getMaskColor(pt,image.image_hsv)
    color_list1.append(color)
print color_list1
subplot,im = getSubplot(edges,im,dist_avg)
for pt in pts:
    G_mask = mask_list[5]
    value_sum = 0
    count = 0
    for i in range(-25,25,1):
        for j in range(-25,25,1):
            value = G_mask[(pt[0]+i,pt[1]+j)]
            value_sum += value
            count += 1
    value_avg = value_sum / count
    print 'value_avg',value_avg
    color = getColor(pt,image.image_hsv)
