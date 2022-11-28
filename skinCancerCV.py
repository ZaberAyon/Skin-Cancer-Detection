import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import math

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


#Read Image
image = cv2.imread('D:\Projects\Skin_cancer\Skin-Cancer-Detection\skinLesion\IMD033.jpg')

main_img = remove_black_shades(image)
#cv2.imshow("OriginalImage", main_img)

hieght, width, ch = main_img.shape
print("Width",width)
print("Hieght",hieght)

# if width>=640 and hieght>=480:
#     image = image_scaling(image)
# cv2.imshow('image',image)
# cv2.imshow("Black Shade Removed",main_img)

#Illumination Correction
lab= cv2.cvtColor(main_img, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=4 ,tileGridSize=(8,8))

cl = clahe.apply(l)


limg = cv2.merge((cl,a,b))

luminance = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

#Hair Removal

removed, canny, close = hair_removal(main_img)

cv2.imshow('HairRemoved', removed)
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
cv2.imshow("opening", thresh)

#Sure background area

sure_bg = cv2.dilate(opening, kernel, iterations=8)
sure_fg = cv2.erode(opening, kernel, iterations=8)

cv2.imshow('surebg', sure_bg)
# cv2.imshow('Erode', sure_bg1)

#Finding sure foreground area

# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,.1*dist_transform.max(), 255, 0)

i, area = major_area_count(sure_fg)

mask = np.zeros_like(sure_fg)
cv2.drawContours(mask, [i], 0, (255,255,255), -1)
cv2.imshow("mask",mask)
#Counting Moments
M = cv2.moments(i)
#Counting Center cordinants
cx = int(M["m10"]/M["m00"])
cy = int(M["m01"]/M["m00"])

q,r,s,t = cv2.boundingRect(i)
print(q,r,s,t)
sure_fg_crop = sure_fg[r:r+t, q:q+s]

cv2.imshow("Cropped",sure_fg_crop)

#Finding sure foreground area

# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,.1*dist_transform.max(), 255, 0)

#cv2.imshow('Sure Foreground', sure_fg)


perimeter = cv2.arcLength(i, True)



W, H  = mask.shape


#Counting Moments
#M = cv2.moments(cnt)
#Counting Center cordinants
#cx = int(M["m10"]/M["m00"])
#cy = int(M["m01"]/M["m00"])
sure_fg = np.uint8(mask)

#Collecting nonzero pixel cordinates of contour
y, x = np.nonzero(sure_fg)
print(y)
print(x)
x = x - np.mean(x)
y = y - np.mean(y)
coords = np.vstack([x, y])

#Convariance
cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)
print(evals)
print(evecs)
#Eigenvector Calculation
sort_indices = np.argsort(evals)[::-1]
evec1, evec2 = evecs[:, sort_indices]
x_v1, y_v1 = evec1
x_v2, y_v2 = evec2

#theta calculation

theta = np.tanh((x_v1)/(y_v1))
theta_dig = theta * 57.2957795 #radian to degree conversion
print("theta in degree", theta_dig)

#Rotate Along mejor and minor axis
#[cos(theta)  -sin(theta)
# sin(theta)  cos(theta)]

MM = cv2.getRotationMatrix2D((cx,cy),theta_dig,1)
cos = np.abs(MM[0,0])
sin = np.abs(MM[0,1])

#minor theta calculation
thetaminor = np.tanh((x_v2)/(y_v2))
theta_dig_min = thetaminor * 57.2957795
print("theta minor in degree", theta_dig_min)


nW = int((H*sin) + (W*cos))
nH = int((H*cos) + (W*sin))

print("axis W", nW)
print("axis H", nH)



MM[0, 2] += (nW / 2) - cx
MM[1, 2] += (nH / 2) - cy

rotate = cv2.warpAffine(sure_fg,MM,(nW,nH))
cv2.imshow("Rotate Major", rotate)

rotate_minor = rotate.copy()
#Raw Rotation Matrix
# theta1 = np.arctan((x_v1)/(y_v1))
# print("theta1",theta1)
# rotation_mat = np.matrix([[np.cos(theta1), -np.sin(theta1)],[np.sin(theta1), np.cos(theta1)]])
# transformed_mat = rotation_mat * coords
# x_transformed, y_transformed = transformed_mat.A
# x_transformed = x_transformed + np.mean(x_transformed)
# y_transformed = y_transformed + np.mean(y_transformed)



#rotate 90 degree
MMX = cv2.getRotationMatrix2D((cx,cy),theta_dig_min,1)
cos = np.abs(MMX[0,0])
sin = np.abs(MMX[0,1])

nWX = int((H*sin) + (W*cos))
nHX = int((H*cos) + (W*sin))

MMX[0, 2] += (nWX / 2) - cx
MMX[1, 2] += (nHX / 2) - cy

rotateX = cv2.warpAffine(sure_fg,MMX,(nWX,nHX))
cv2.imshow("RO",rotateX)

#along major_axis
cropped_top = rotate_minor[0:int(nH*.5) , 0:int(nW)]
cropped_bot = rotate_minor[int(nH*.5):int(nH) , 0:int(nW)]
mirror_minor = cv2.flip(cropped_top,0)
cv2.imshow("Flipped_minor",mirror_minor)

#Translation
G = np.float32([[1,0,0],[0,1,nH*.5]])
res_top = cv2.warpAffine(mirror_minor,G,(nW,nH))
cv2.imshow("Flipped Top",res_top)

L = np.float32([[1,0,0],[0,1,nH*.5]])
res_bot = cv2.warpAffine(cropped_bot,L,(nW,int(nH)))
cv2.imshow("Bottom_minor",res_bot)

subtracted_minor = cv2.subtract(res_top, res_bot)
cv2.imshow("Subtracted minor",subtracted_minor)
cv2.imshow("Cropped Top Minor", cropped_top)
cv2.imshow("Cropped Bottom Minor",cropped_bot)


#along minor axis


top = rotateX[0:int(nHX*.5) , 0:int(nWX)]
bottom = rotateX[int(nHX*.5):int(nHX) , 0:int(nWX)]
major = cv2.flip(top,0)

G = np.float32([[1,0,0],[0,1,nHX*.5]])
res_top_major = cv2.warpAffine(major,G,(nWX,int(nHX)))
#cv2.imshow("flipped Top major",res_top_major)

L = np.float32([[1,0,0],[0,1,nHX*.5]])
res_bot_major = cv2.warpAffine(bottom,L,(nWX,int(nHX)))
#cv2.imshow("bottom_major",res_bot_major)
subtracted_major = cv2.subtract(res_top_major,res_bot_major)

area_minor = area_count(subtracted_minor)

area_major = area_count(subtracted_major)

A_minor = (area - area_minor)/area

A_major = (area - area_major)/area


print(A_minor)
print(A_major)


print("Parimeter")
print(perimeter)
print("Area")
print(area)

has = 0
vas = 0
#Asymmetry
#A = 0
if(A_minor<=0.90):
    vas = 1
if(A_major<=0.90):
    has = 1

A = has + vas
print("Asymmetry")
print(A)
# #Finding unknown region

# sure_fg = np.uint8(sure_fg)

# cv2.imshow("Foreground",sure_fg)
# cv2.imshow("BackGround",sure_bg)

#Border_irregularity

#Harris-Stephens algorithm
gray = np.float32(sure_fg)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

res = np.hstack((centroids,corners))
res = np.int0(res)

print("Corner No.")
print(len(res))
#print res


maxi = res.max()
mini = res.min()

#print maxi
#print mini
#print m

B = (4*np.pi*area)/(perimeter*perimeter)

# B = int(len(res)/8)

# if len(res)>150:
#     B = B+1

print("Border Irregularity")
print(B)

#Diameter

(x,y), radius = cv2.minEnclosingCircle(i)

#diameter in cm
temp = math.sqrt(4*area/math.pi)
#d = (radius*2)*0.0264583333333334
d = (temp * 25.4) / 96
D = (d/20)
print("Diameter")
print(d)


# if d<=1.5:
#     D = 1
# elif d>1.5 and d<2.5:
#     D = 2
# elif d>=2.5 and d<3.5:
#     D = 3
# elif d>=3.5 and d<=4.5:
#     D = 4
# elif d>4.5:
#     D = 5

print("Diameter Score")
print(D)

#Minimum Circle Drawing

#center =(int(x),int(y))
#rad = int(radius)
#cv2.circle(image,center,rad,(0,255,0),2)

#Color index

x,y,w,h = cv2.boundingRect(i)
crop = removed[y:y+h, x:x+w]

#k-means clustering

Z= crop.reshape(-1,3)

Z= np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K=6
counts = 0
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res=center[label.flatten()]
res2=res.reshape((crop.shape))
for clr in center:
    for n, val in enumerate(clr):
        globals()["var%d"%n] = val
    print(var0)
    print(var1)
    print(var2)

    if (var0<=52 and var1<=52 and var2<=62):
        counts=counts+1
    elif(var0>=205 and var1>=205 and var2>=205):
        counts=counts+1
    elif(var0<52 and var1<52 and var2>=150):
        counts=counts +1
    elif((var0>= 0 and var0<=100) and (var1>50 and var1<=150) and (var2>=150 and var2<=240)):
        counts=counts+1
    elif( (var0>0 and var0<100) and (var1>=0 and var1<100) and (var2>62 and var2<150)):
        counts=counts+1
    elif((var0>=125 and var0<=150) and (var1>=100 and var1<=125) and (var2>=0 and var2<=150)):
        counts=counts+1

C = counts
print("Color Index")
print(C)


#TDS Calculation
tds = 1.3*A + 0.1*B + 0.5*C + 0.5*D

print("TDS Value")
print(tds)

if tds>5.45:
    print('Malignant')

else:
    print('Benign')

#cv2.imshow('Thresh', thresh)
#cv2.imshow('Thres', opening)
#cv2.imshow('Thres', dist_transform)
#cv2.imshow("kmeans",res2)
# cv2.imshow("LAB", lab)
# cv2.imshow('Labnew',limg)
# cv2.imshow('Luminance', luminance)

# cv2.imshow('Opening', opening)
# cv2.imshow('Closing',closing)


# cv2.imshow('Unknown', unknown)
# cv2.imshow('Crop', crop)
font = cv2.FONT_HERSHEY_SIMPLEX
if(tds>5.45):
    cv2.putText(removed, 'MALIGNANT', (260,30), font, 1, (0,0,255),1,cv2.LINE_AA)

else:
    cv2.putText(removed, 'BENIGN', (270,30), font, 1, (0,0,255),1,cv2.LINE_AA)
cv2.imshow('Result', removed)

W, H  = sure_fg.shape

unknown = cv2.subtract(sure_bg, sure_fg)
#cv2.imshow("unknown",unknown)

# Marker labelling

ret, markers = cv2.connectedComponents(sure_fg)


markers = markers+1


# Mark the region of unknown with zero

markers[unknown==255] = 0


#watershed

markers = cv2.watershed(removed, markers)


removed[markers == -1] = [255, 0, 0]

#cv2.imshow("WaterShed Segmentation", removed)
cv2.waitKey(0)
cv2.destroyAllWindows()