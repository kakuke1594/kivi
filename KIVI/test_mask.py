import cv2
import numpy as np

img1 = cv2.imread('img/ma_de.jpg')
img = cv2.imread('img/ma_de.jpg',0)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Create mask
height ,width = img.shape
mask = np.zeros((height, width), np.uint8)

# edges = cv2.Canny(thresh, 100, 200)
cv2.imshow('detected ', gray)
cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=7.1, minDist=10,
                           param1=200, param2=15,
                           minRadius=1, maxRadius=10)
for i in circles[0, :]:
    i[2]=i[2]+4
    # Draw on mask
    cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)

# Copy that image using that mask
masked_data = cv2.bitwise_and(img1, img1, mask=mask)

# Apply Threshold
_,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

# Find Contour
contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(contours[0])

# Crop masked_data
crop = masked_data[y:y+h, x:x+w]

#Code to close Window
cv2.imshow('detected Edge',img1)
cv2.imshow('Cropped Eye',crop)
cv2.waitKey(0)
cv2.destroyAllWindows()