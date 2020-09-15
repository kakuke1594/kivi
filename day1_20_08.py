# import cv2
# import numpy as np
#
# path = "IMG_6009.JPG"
# img = cv2.imread(path)
# img = cv2.resize(img, (800, 800))
# kernel = np.ones((5, 5), np.uint8)
#
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# # img_blurr = cv2.GaussianBlur(img_gray, (5,5), 0)
# # img_canny = cv2.Canny(img, 200, 200)
# # img_dialation = cv2.dilate(img_canny, kernel, iterations=5)
# #
# # cv2.imshow("Picture Gray", img_gray)
# # cv2.imshow("Picture Blur", img_blurr)
# # cv2.imshow("Picture Canny", img_canny)
# # cv2.imshow("Picture Dialation", img_dialation)
# mask = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255,30]))
# cv2.imshow("HSV", mask)
# cv2.waitKey(0)
#
# ---------------------------------------------
# import cv2
# import numpy as np
#
# ##(1) read into  bgr-space
# img = cv2.imread("IMG_6011.JPG")
# img = cv2.resize(img, (800, 800))
# ##(2) convert to hsv-space, then split the channels
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)
#
# ##(3) threshold the S channel using adaptive method(`THRESH_OTSU`) or fixed thresh
# th, threshed = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
#
# ##(4) find all the external contours on the threshed S
# #_, cnts, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#
# canvas  = img.copy()
# #cv2.drawContours(canvas, cnts, -1, (0,255,0), 1)
#
# ## sort and choose the largest contour
# cnts = sorted(cnts, key = cv2.contourArea)
# cnt = cnts[-1]
#
# ## approx the contour, so the get the corner points
# arclen = cv2.arcLength(cnt, True)
# approx = cv2.approxPolyDP(cnt, 0.02* arclen, True)
# cv2.drawContours(canvas, [cnt], -1, (255,0,0), 1, cv2.LINE_AA)
# cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 1, cv2.LINE_AA)
#
# ## Ok, you can see the result as tag(6)
# # cv2.imwrite("detected.png", canvas)
# cv2.imshow("dd", canvas)
# cv2.waitKey(0)

# ---------------------------------------------
from imutils.perspective import four_point_transform
import cv2
import numpy as np


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



# Load image, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread("IMG_6011.JPG")
image =image_resize(image, 800, 800)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Find contours and sort for largest contour
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

for c in cnts:
    # Perform contour approximation
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        displayCnt = approx
        break

# Obtain birds' eye view of image
warped = four_point_transform(image, displayCnt.reshape(4, 2))



cv2.imshow("thresh", thresh)
cv2.imwrite("thresh.png", thresh)
cv2.imshow("warped", warped)
cv2.imwrite("warped.png", warped)
# mask = cv2.inRange(warped, np.array([0, 0, 0]), np.array([180, 255,30]))
mask = cv2.inRange(warped, (0, 0, 0, 0), (180, 255, 70, 0))
# cnt = cnts[4]
# draw = cv2.drawContours(warped, [cnts], 0, (0, 255, 0), 3)
cv2.imshow("HSV", mask)
cv2.imwrite("HSV.png", mask)


cv2.waitKey(0)
