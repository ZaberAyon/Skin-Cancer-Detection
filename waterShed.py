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
def major_area_count(image):

    cntr_frame, p, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = p[0]
    area = cv2.contourArea(cnt)
    area =0
    for i in p:
        if(area>cv2.contourArea(i)):
           cnt=i
    return cnt 


#Read Image
image = cv2.imread('melanoma.png')

#cv2.imshow("OriginalImage", image) 

width, hieght, ch = image.shape

if width>=640 and hieght>=480:
    image = image_scaling(image)
#cv2.imshow('image',image)

#Illumination Correction
lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=4 ,tileGridSize=(8,8))

cl = clahe.apply(l)


limg = cv2.merge((cl,a,b))

luminance = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

#Hair Removal

removed, canny, close = hair_removal(image)

cv2.imshow('HairRemoved', removed)
#cv2.imshow('Canny', canny)
#cv2.imshow('Closing', close)


#Histogram Calculation
gray = cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY)

equ = cv2.equalizeHist(gray)

#Original_image

#hist,bins = np.histogram(image.ravel(),256,[0,256])


#plt.hist(image.ravel(),256,[0,256])
#plt.show()

#Corrected Luminance image

#hist,bins = np.histogram(luminance.ravel(),256,[0,256])

#plt.hist(luminance.ravel(),256,[0,256])
#plt.show()

#Histrogram Equalization

#hist,bins = np.histogram(equ.ravel(),256,[0,256])

#plt.hist(equ.ravel(),256,[0,256])
#plt.show()


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

#cv2.imshow('Sure Foreground', sure_fg)


#Finding unknown region

sure_fg = np.uint8(sure_fg)

W, H  = sure_fg.shape

i = major_area_count(sure_fg)

#Counting Moments
M = cv2.moments(i)
#Counting Center cordinants
cx = int(M["m10"]/M["m00"])
cy = int(M["m01"]/M["m00"])

q,r,s,t = cv2.boundingRect(i)

sure_fg_crop = sure_fg[r:r+t, q:q+s]

S = np.float32([[1,0,(W*.5-cx)],[0,1,(H*.5-cy)]])
sure_fg_resized = cv2.warpAffine(sure_fg_crop,S,(H,W))

cv2.imshow("Resized",sure_fg_resized)

cv2.imshow("Sure_fg_crop",sure_fg_crop)

unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling

ret, markers = cv2.connectedComponents(sure_fg)


markers = markers+1


# Mark the region of unknown with zero

markers[unknown==255] = 0


#watershed

markers = cv2.watershed(removed, markers)


removed[markers == -1] = [255, 0, 0]

#contour detection

cntr_frame, p, hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cnt = p[0]
area = cv2.contourArea(cnt)
for i in p:
    if cv2.contourArea(i) > area:
        cnt = i

        area = cv2.contourArea(i)

perimeter = cv2.arcLength(cnt, True)


#W, H  = sure_fg.shape

#Counting Moments
#M = cv2.moments(cnt)
#Counting Center cordinants
#cx = int(M["m10"]/M["m00"])
#cy = int(M["m01"]/M["m00"])

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

rotate_minor = rotate.copy()

#rotate 90 degree
MMX = cv2.getRotationMatrix2D((nW/2,nH/2),90,1)
cos = np.abs(MMX[0,0])
sin = np.abs(MMX[0,1])

nWX = int((nH*sin) + (nW*cos))
nHX = int((nH*cos) + (nW*sin))

MMX[0, 2] += (nWX / 2) - nW/2
MMX[1, 2] += (nHX / 2) - nH/2

rotateX = cv2.warpAffine(rotate,MMX,(nWX,nHX))
#cv2.imshow("RO",rotateX)

#along minor_axis
cropped_top = rotate_minor[0:int(nH*.5) , 0:int(nW)]
cropped_bot = rotate_minor[int(nH*.5):int(nH) , 0:int(nW)]
mirror_minor = cv2.flip(cropped_top,0)
#cv2.imshow("Flipped_minor",mirror_minor)

#Translation
G = np.float32([[1,0,0],[0,1,nH*.5]])
res_top = cv2.warpAffine(mirror_minor,G,(nW,nH))
#cv2.imshow("Flipped Top",res_top)

L = np.float32([[1,0,0],[0,1,nH*.5]])
res_bot = cv2.warpAffine(cropped_bot,L,(nW,int(nH)))
#cv2.imshow("Bottom_minor",res_bot)

subtracted_minor = cv2.subtract(res_top, res_bot) 
#cv2.imshow("Subtracted minor",subtracted_minor)
#cv2.imshow("Cropped Top Minor", cropped_top) 
#cv2.imshow("Cropped Bottom Minor",cropped_bot)


#along mejor axis

top = rotateX[0:int(nH*.5) , 0:int(nW)]
bottom = rotateX[int(nH*.5):int(nH) , 0:int(nW)]
major = cv2.flip(top,0)

G = np.float32([[1,0,0],[0,1,nH*.5]])
res_top_major = cv2.warpAffine(major,G,(nW,int(nH)))
#cv2.imshow("flipped Top major",res_top_major)

L = np.float32([[1,0,0],[0,1,nH*.5]])
res_bot_major = cv2.warpAffine(bottom,L,(nW,int(nH)))
#cv2.imshow("bottom_major",res_bot_major)


#cv2.imshow("Flipped_major",major)
subtracted_major = cv2.subtract(res_top_major,res_bot_major)
#cv2.imshow("Subtracted major",subtracted_major)
#cv2.imshow("Cropped Top Major", top) 




area_minor = area_count(subtracted_minor)

area_major = area_count(subtracted_major)

cv2.imshow("Subtracted major",subtracted_major)


print "Area according to minor axis"
print area_minor

print "Area according to major axis"
print area_major

A_minor = (area_minor/area)*100

A_major = (area_major/area)*100

print A_minor
print A_major


print "Parimeter"    
print perimeter
print "Area"
print area

#Asymmetry  
#A = 0
if A_minor>=10 and A_minor<=25:
    A = 1
elif A_major>=10 and A_major<=25:
    A = 1
elif (A_minor>=10 and A_minor<=25) and (A_major>=10 and A_major<=25):
    A = 2
else:
    A = 0
print "Asymmetry"
print A
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

print "Corner No."
print len(res)
#print res


maxi = res.max()
mini = res.min()

#print maxi
#print mini
#print m

B = int(len(res)/8)

if len(res)>150:
    B = B+1

print "Border Irregularity"
print B

#Diameter

(x,y), radius = cv2.minEnclosingCircle(cnt)

#diameter in cm

d = (radius*2)*0.0264583333333334

print "Diameter in cm"
print d

if d<=1.5:
    D = 1
elif d>1.5 and d<2.5:
    D = 2
elif d>=2.5 and d<3.5:
    D = 3
elif d>=3.5 and d<=4.5:
    D = 4
elif d>4.5:
    D = 5

print "Diameter Score"
print D

#Minimum Circle Drawing

#center =(int(x),int(y))
#rad = int(radius)
#cv2.circle(image,center,rad,(0,255,0),2)

#Color index

x,y,w,h = cv2.boundingRect(cnt)
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
    print var0
    print var1
    print var2

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
print "Color Index"
print C


#TDS Calculation
tds = 1.3*A + 0.1*B + 0.5*C + 0.5*D

print "TDS Value"
print tds

if tds>5.45:
    print 'Malignant'

else:
    print 'Benign'

#cv2.imshow('Thresh', thresh)
#cv2.imshow('Thres', opening)
#cv2.imshow('Thres', dist_transform)
#cv2.imshow("kmeans",res2)
cv2.imshow("LAB", lab)
cv2.imshow('Labnew',limg)
cv2.imshow('Luminance', luminance)

#cv2.imshow('Opening', opening)
#cv2.imshow('Closing',closing)


#cv2.imshow('Unknown', unknown)
#cv2.imshow('Crop', crop)
font = cv2.FONT_HERSHEY_SIMPLEX
if(tds>5.9):
    cv2.putText(removed, 'MALIGNANT', (260,30), font, 1, (0,0,255),1,cv2.LINE_AA)

else:
    cv2.putText(removed, 'BENIGN', (270,30), font, 1, (0,0,255),1,cv2.LINE_AA)
cv2.imshow('Result', removed)

cv2.waitKey(0)
cv2.destroyAllWindows()
