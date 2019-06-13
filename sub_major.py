import numpy as np
import cv2
import math
#from matplotlib import pyplot as plt

def image_scaling(image):
    
    newimage = cv2.resize(image,(520,380))
    return newimage   

def area_count(image):
    cntr_frame, p, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cnt = p[0]
    #area = cv2.contourArea(cnt)
    area =0
    for i in p:

        area1 = cv2.contourArea(i)

        area = area+area1

    return area

def hair_removal(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray,100,200)

    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)
    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations = 3)

    hairRemoved = cv2.inpaint(image,closing,3,cv2.INPAINT_TELEA)

    return hairRemoved, canny, closing

def largest_area_count(image):
    cntr_frame, p, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt_X = p[0]
    area_X = cv2.contourArea(cnt_X)
    for i in p:
        if cv2.contourArea(i) > area_X:
            cnt_X = i

    return cnt_X

#Read Image
image = cv2.imread('melanoma.png')

#cv2.imshow("OriginalImage", image) 

width, hieght, ch = image.shape

if width>=640 and hieght>=480:
    image = image_scaling(image)
#cv2.imshow('image',image)

removed, canny, close = hair_removal(image)
gray = cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY)


#Filterring

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


#Morfological Transformation

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)

#closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 3)


#Sure background area

sure_bg = cv2.dilate(opening, kernel, iterations=8)
#sure_bg1 = cv2.erode(opening, kernel, iterations=8)

#cv2.imshow('surebg', sure_bg)
#cv2.imshow('Erode', sure_bg1)


#Finding sure foreground area

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,.1*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)

#cv2.imshow('Sure Foreground', sure_fg)

cntr_frame, p, hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cnt = p[0]
area = cv2.contourArea(cnt)
for i in p:
    if cv2.contourArea(i) > area:
        cnt = i

        area = cv2.contourArea(i)

perimeter = cv2.arcLength(cnt, True)


W, H  = sure_fg.shape

#Counting Moments
M = cv2.moments(cnt)
#Counting Center cordinants
cx = int(M["m10"]/M["m00"])
cy = int(M["m01"]/M["m00"])

#Collecting nonzero pixel cordinates of contour
y, x = np.nonzero(sure_fg)
x = x - np.mean(x)
y = y - np.mean(y)
coords = np.vstack([x, y])

#Convariance
cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)

#Eigenvector Calculation
sort_indices = np.argsort(evals)[::-1]
evec1, evec2 = evecs[:, sort_indices]
x_v1, y_v1 = evec1
x_v2, y_v2 = evec2

#theta calculation

theta = np.tanh((x_v1)/(y_v1))
theta_dig = theta * 57.2957795 #radian to degree conversion

#Rotate Along mejor and minor axis

MM = cv2.getRotationMatrix2D((cx,cy),theta_dig,1)
cos = np.abs(MM[0,0])
sin = np.abs(MM[0,1])

nW = int((H*sin) + (W*cos))
nH = int((H*cos) + (W*sin))

MM[0, 2] += (nW / 2) - cx
MM[1, 2] += (nH / 2) - cy

rotate = cv2.warpAffine(sure_fg,MM,(nW,nH))

#rotate 90 degree
MMX = cv2.getRotationMatrix2D((nW/2,nH/2),90,1)
cos = np.abs(MMX[0,0])
sin = np.abs(MMX[0,1])

nWX = int((nH*sin) + (nW*cos))
nHX = int((nH*cos) + (nW*sin))

MMX[0, 2] += (nWX / 2) - nW/2
MMX[1, 2] += (nHX / 2) - nH/2

rotateX = cv2.warpAffine(rotate,MMX,(nWX,nHX))
cv2.imshow("RO",rotateX)

#along mejor axis

top = rotateX[0:int(nH*.5) , 0:int(nW)]
bottom = rotateX[int(nH*.5):int(nH) , 0:int(nW)]
major = cv2.flip(top,0)

G = np.float32([[1,0,0],[0,1,nH*.5]])
res1 = cv2.warpAffine(major,G,(nW,int(nH)))
cv2.imshow("RES2",res1)

L = np.float32([[1,0,0],[0,1,nH*.5]])
res2 = cv2.warpAffine(bottom,L,(nW,int(nH)))
cv2.imshow("RES",res2)


cv2.imshow("Flipped_major",major)
subtracted_major = cv2.subtract(res1,res2)
cv2.imshow("Subtracted major",subtracted_major)
cv2.imshow("Cropped Top Major", top) 


cv2.waitKey(0)
cv2.destroyAllWindows()


