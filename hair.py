import cv2
import numpy as np



def hair_removal(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray,100,200)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)
    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations = 3)

    hairRemoved = cv2.inpaint(image,opening,3,cv2.INPAINT_TELEA)

    return hairRemoved, canny, closing

image = cv2.imread('melanoma.png')

removed, canny, closing = hair_removal(image)

cv2.imshow("Removed", removed)
cv2.imshow("Closing", closing)


cv2.waitKey(0)
cv2.destroyAllWindows()
