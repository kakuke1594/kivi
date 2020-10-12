import cv2

img = cv2.imread('second.jpg')
y = 18
x = 272
h = 160
w = 100
crop_img = img[y:y+h, x:x+w]
cv2.imshow("MSSV", crop_img)
cv2.waitKey(0 )