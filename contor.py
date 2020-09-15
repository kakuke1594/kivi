import cv2
import numpy as np
from utils import stackImages

def getContour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 2)

        if area > 1350:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.95 and aspRatio < 1.05:
                    objType="Rectangle"
            else: objType =''
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(imgContour, objType, (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)



path = "IMG_6009.JPG"
img = cv2.imread(path)
img = cv2.resize(img, (600, 600))
imgContour = img.copy()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

img_blur = cv2.GaussianBlur(img_gray, (7, 7), 1)
img_canny = cv2.Canny(img_blur, 200, 200)
getContour(img_canny)

img_black = np.zeros_like(img)
img_stack = stackImages(([img, img_gray, img_blur],
                         [img_canny, imgContour, img_black]), 0.6)
cv2.imshow("stack image", img_stack)
cv2.waitKey(0)


