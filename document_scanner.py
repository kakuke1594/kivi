import cv2

img = cv2.imread("IMG_6003.JPG")
ratio = img.shape[0]/ 500.0
orig = img.copy()
