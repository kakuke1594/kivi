import cv2
import numpy as np

path = "IMG_6003.JPG"
img = cv2.imread(path)
img = cv2.resize(img, (800, 800))

img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
img_blurr = cv2.GaussianBlur(img_luv, (5,5), 0)
img_canny = cv2.Canny(img, 200, 200)
# img_dialation = cv2.dilate(img_canny, kernel, iterations=5)
cv2.imshow("img", img_luv)
cv2.imshow("luv", img_canny)

cv2.waitKey(0)