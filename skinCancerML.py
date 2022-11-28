import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import math
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from skimage import util, exposure
from skimage.filters import sobel
import os
import pandas as pd

def image_scaling(image):

    newimage = cv2.resize(image,(520,380))
    return newimage

def area_count(image):
    p, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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
def major_area_count(image):

    p, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = p[0]
    area = cv2.contourArea(cnt)
    for i in p:
        temp_area = cv2.contourArea(i)
        if(temp_area>area):
            area = temp_area
            cnt = i
    return cnt, area


#Remove black shades from ph2 dataset darmatoscopy images
def remove_black_shades(image):
    h, w, c = image.shape
    diameter = 0
    if(w<h):
        diameter = w
    elif(h<w):
        diameter = h
    else:
        diameter = h


    center = (w/2, h/2)
    angle = 0
    points = cv2.boxPoints((center,(diameter,diameter),angle))
    #print(points)
    x1,y1,x2,y2,x3,y3,x4,y4 = points[0][0],points[0][1],points[1][0],points[1][1],points[2][0],points[2][1],points[3][0],points[3][1]
    #print(y1,y3,x1,x2)
    cropped_main_img = image[int(y2):int(y1),int(x2):int(x3)]
    return cropped_main_img
def image_statistics(Z):
    #Input: Z, a 2D array, hopefully containing some sort of peak
    #Output: cx,cy,sx,sy,skx,sky,kx,ky
    #cx and cy are the coordinates of the centroid
    #sx and sy are the stardard deviation in the x and y directions
    #skx and sky are the skewness in the x and y directions
    #kx and ky are the Kurtosis in the x and y directions
    #Note: this is not the excess kurtosis. For a normal distribution
    #you expect the kurtosis will be 3.0. Just subtract 3 to get the
    #excess kurtosis.
    import numpy as np

    h,w = np.shape(Z)

    x = range(w)
    y = range(h)


    #calculate projections along the x and y axes
    yp = np.sum(Z,axis=1)
    xp = np.sum(Z,axis=0)

    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)

    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2

    sx = np.sqrt( np.sum(x2*xp)/np.sum(xp) )
    sy = np.sqrt( np.sum(y2*yp)/np.sum(yp) )

    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3

    skx = np.sum(xp*x3)/(np.sum(xp) * sx**3)
    sky = np.sum(yp*y3)/(np.sum(yp) * sy**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    kx = np.sum(xp*x4)/(np.sum(xp) * sx**4)
    ky = np.sum(yp*y4)/(np.sum(yp) * sy**4)


    return cx,cy,sx,sy,skx,sky,kx,ky



df = pd.read_csv("D:\Projects\Skin_cancer\Skin-Cancer-Detection\polished.csv")
filenames = df['filename'].to_list()
a_minor = df['a_minor'].to_list()
a_major = df['a_major'].to_list()
borderirr = df['borderirr'].to_list()
colorind = df['colorind'].to_list()
diam = df['diameter'].to_list()
target = df['result'].to_list()

energyL = []
stdL = []
kxL = []
kyL = []
skxL = []
skyL = []
contrastL = []
dissimilarityL = []
homogeneityL = []
correlationL = []
meanL = []

folder = 'D:\Projects\Skin_cancer\Skin-Cancer-Detection\skinLesion'
for filename in os.listdir(folder):
#Read Image

    image = cv2.imread(os.path.join(folder,filename))

    main_img = remove_black_shades(image)
    #cv2.imshow("OriginalImage", main_img)

    hieght, width, ch = main_img.shape
    print("Width",width)
    print("Hieght",hieght)

    # if width>=640 and hieght>=480:
    #     image = image_scaling(image)
    # cv2.imshow('image',image)

    #Illumination Correction
    lab= cv2.cvtColor(main_img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=4 ,tileGridSize=(8,8))

    cl = clahe.apply(l)


    limg = cv2.merge((cl,a,b))

    luminance = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    #Hair Removal

    removed, canny, close = hair_removal(main_img)

    #cv2.imshow('HairRemoved', removed)
    #cv2.imshow("luminance",luminance)
    # cv2.imshow("canny",canny)
    # cv2.imshow("closing",close)

    #Histogram Calculation
    gray = cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    equ = cv2.equalizeHist(gray)

    #Filterring

    ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    #Morfological Transformation

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)

    #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 3)
    #cv2.imshow("opening", thresh)

    #Sure background area

    sure_bg = cv2.dilate(opening, kernel, iterations=8)
    sure_fg = cv2.erode(opening, kernel, iterations=8)

    #cv2.imshow('surebg', sure_bg)
    #cv2.imshow('Erode', sure_bg1)

    #Finding sure foreground area

    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,.1*dist_transform.max(), 255, 0)

    i, area = major_area_count(sure_fg)

    mask = np.zeros_like(sure_fg)
    cv2.drawContours(mask, [i], 0, (255,255,255), -1)
    #Counting Moments

    #Counting Moments
    M = cv2.moments(i)
    #Counting Center cordinants
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    q,r,s,t = cv2.boundingRect(i)
    print(q,r,s,t)
    sure_fg_crop = mask[r:r+t, q:q+s]
    sure_rgb = removed[r:r+t, q:q+s]
    W, H  = mask.shape
    sure_rgb_t = np.uint8(sure_rgb)

    mask_inv = cv2.bitwise_not(sure_fg_crop)



    sure_rgb_f = cv2.bitwise_and(sure_rgb_t,sure_rgb_t,sure_fg_crop)
    sure_fg_gray = cv2.cvtColor(sure_rgb_f, cv2.COLOR_BGR2GRAY)

    #Counting Moments
    #M = cv2.moments(cnt)
    #Counting Center cordinants
    #cx = int(M["m10"]/M["m00"])
    #cy = int(M["m01"]/M["m00"])

    sure_fg = np.uint8(mask)

    y, x = np.nonzero(sure_fg)
    print(y)
    print(x)
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

    glcm = greycomatrix(sure_fg_gray, [10], [0, theta], levels=256)

    #Contrast, energy, correlation, dissimilarity, homogeneity
    contrast = greycoprops(glcm, 'contrast')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    contrastL.append(contrast[0][0])
    energyL.append(energy[0][0])
    correlationL.append(correlation[0][0])
    dissimilarityL.append(dissimilarity[0][0])
    homogeneityL.append(homogeneity[0][0])

    #std and mean
    std=np.nanstd(np.where(np.isclose(sure_rgb_f,0), np.nan, sure_rgb_f))


    np.seterr(divide='ignore', invalid='ignore')
    average = np.true_divide(sure_rgb_f.sum(1),(sure_rgb_f!=0).sum(1))
    mean = np.mean(average)

    stdL.append(std)
    meanL.append(mean)
    print(mean)



    ## skewness and courtosis and centroid


    cx,cy,sx,sy,skx,sky,kx,ky = image_statistics(sure_fg_gray)

    skxL.append(skx)
    skyL.append(sky)
    kxL.append(kx)
    kyL.append(ky)

dataFrame = pd.DataFrame(list(zip(filenames,a_minor,a_major,borderirr,colorind,diam,meanL,stdL,energyL,homogeneityL,contrastL,dissimilarityL,skxL,skyL,kxL,kyL,target)),columns=['filename','a_minor','a_major','borderirr','colorind','diameter','mean','std','energy','homogenity','contrast','dissimilarity','kurtosisx','kurtosisy','skewnessx','skewnessy','target'])
dataFrame.to_csv("final.csv")
cv2.waitKey(0)
cv2.destroyAllWindows()