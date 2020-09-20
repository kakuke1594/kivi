from imutils.perspective import four_point_transform, order_points
import cv2
from matplotlib import pyplot as plt
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_4_corner(pts):
    top_left = list()
    top_right = list()
    bot_right = list()
    bot_left = list()
    for p in pts:
        rect = order_points(p)
        (tl, tr, br, bl) = rect
        top_left.append(tl.tolist())
        top_right.append(tr.tolist())
        bot_right.append(br.tolist())
        bot_left.append(bl.tolist())
    top_left = sorted(top_left, key=lambda x: (x[0], x[1]))
    top_right = sorted(top_right, key=lambda x: (x[0], x[1]))
    bot_left = sorted(bot_left, key=lambda x: (x[0], x[1]))
    bot_right = sorted(bot_right, key=lambda x: (x[0], x[1]))
    return [top_left[0], top_right[2], bot_right[3], bot_left[1]]



# Load image, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread("IMG_6011.JPG")
image = image_resize(image, 800, 800)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Find contours and sort for largest contour
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        displayCnt = approx
        break

# Obtain birds' eye view of image
warped = four_point_transform(image, displayCnt.reshape(4, 2))

# # cv2.imshow("thresh", thresh)
# # cv2.imwrite("thresh.png", thresh)
# # cv2.imshow("warped", warped)
# # cv2.imwrite("warped.png", warped)
# # mask = cv2.inRange(warped, np.array([0, 0, 0]), np.array([180, 255,30]))
# mask = cv2.inRange(warped, (0, 0, 0, 0), (180, 255, 70, 0))
# # cnt = cnts[4]
# # draw = cv2.drawContours(warped, [cnts], 0, (0, 255, 0), 3)
# cv2.imshow("HSV", mask)
# cv2.imwrite("HSV.png", mask)


# cv2.waitKey(0)


gray = cv2.cvtColor(warped.copy(), cv2.COLOR_BGR2GRAY)
# gray = cv2.blur(gray, (11, 11))
ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
img, contour, hierrachy = cv2.findContours(thresh, 1, 2)
black_point = list()
boundary = list()
for c, cnt in enumerate(contour):
    area = cv2.contourArea(cnt)
    if area > 200:
        cv2.drawContours(warped, cnt, -1, (255, 255, 0), 3)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        black_point.append(approx.reshape(4, 2))
        objCor = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        boundary.append((x, y, w, h))
        cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)

a = get_4_corner(black_point)
# count = np.asarray(boundary)
# max_height = np.max(count[::, 3])
# nearest = max_height * 1.4
# boundary.sort(key=lambda r: [int(nearest * round(float(r[1]) / nearest)), r[0]])
# aaaaa = list()
#
# for i in boundary:
#     bbbb = list()
#     for idx, val in enumerate(i):
#         if idx < 2:
#             bbbb.append(val)
#     aaaaa.append(bbbb)

a = np.array(a)
second = four_point_transform(warped, a)
# mask = cv2.inRange(second, (0, 0, 0, 0), (180, 255, 70, 0))

plt.imshow(warped)
plt.imshow(second)
# plt.imshow(mask)
plt.show()
cv2.waitKey(0)
