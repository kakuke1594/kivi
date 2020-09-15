import cv2

import numpy as np

img = cv2.imread('IMG_6003.JPG')
img_resize = cv2.resize(img, (800, 800))
img_cropped = img[0:400, 400:800]

cv2.imshow('Image', img_resize)
cv2.imshow('Image crop', img_cropped)

cv2.waitKey(0)