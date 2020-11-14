from imutils.perspective import four_point_transform, order_points
import cv2
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


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


def draw_rectangle(img, pts_1, pst_2, h, w):
    x = pts_1.tolist()[0]
    cv2.rectangle(img, (x[0][0], x[0][1]), (x[0][0] + w, x[0][1] + h), 3)
    crop_img = img[x[0][1]: x[0][1] + h,x[0][0]: x[0][0] +w]
    return crop_img


def split_boxs(img):
    rows  = np.hsplit(img, 5)
    cv2.imshow("Split", rows[0])
    cv2.imshow("Split 2", rows[1])


def get_circles(mssv, img):
    img_cnt, cnt, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnt:
        are = cv2.contourArea(c)
        if are > 10:
            cv2.drawContours(mssv, c, -1, (255, 0, 255), 1)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor > 4: obj = 'Circle'
            cv2.rectangle(mssv, (x, y), (x+w, y+h), (0,255,0), 2)


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

# Get 4 big square
gray = cv2.cvtColor(warped.copy(), cv2.COLOR_BGR2GRAY)
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
a = np.array(a)
second = four_point_transform(warped, a)

# get small square
mask = cv2.inRange(second, (0, 0, 0, 0), (180, 255, 70, 0))

_, cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

mini_square = list()
mini_bouding = list()
mini_cont = list()
for ca in cont:
    area = cv2.contourArea(ca)
    if 40 < area < 100:
        cv2.drawContours(second, ca, -1, (255, 0, 255), 1)
        peri = cv2.arcLength(ca, True)
        approx = cv2.approxPolyDP(ca, 0.02 * peri, True)
        mini_square.append(approx)
        mini_cont.append(ca)
        x, y, w, h = cv2.boundingRect(approx)
        mini_bouding.append((x, y, w, h))

mini_cont.sort(key=lambda x: get_contour_precedence(x, second.shape[1]))
# for i in range(len(mini_cont)):
#     second = cv2.putText(second, str(i), cv2.boundingRect(mini_cont[i])[:2], cv2.FONT_ITALIC, 1, [125])



new_pos = dict()
cols = defaultdict(list)
y_values = list()

y_value = None
col = 1
for ca in mini_cont:
    area = cv2.contourArea(ca)
    if 40 < area < 100:
        peri = cv2.arcLength(ca, True)
        approx = cv2.approxPolyDP(ca, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        new_pos[(x, y, w, h)] = ca
        if y_value is None:
            y_value = y
            cols[col].append((x, y, w, h))
            continue
        elif y_value - 2 < y < y_value + 2:
            cols[col].append((x, y, w, h))
        else:
            col += 1
            cols[col].append((x, y, w, h))
            y_value = y

list_of_contour = defaultdict(list)
for column in cols:
    for c in cols[column]:
        # second = cv2.putText(second, str(column), cv2.boundingRect(new_pos.get(c))[:2], cv2.FONT_ITALIC, 1, [125])
        list_of_contour[column].append(new_pos.get(c))

# draw rectangle
blocks = list()
mssv = None
ma_de = None
for key in list(list_of_contour):
    if key == 1:
        for idx, val in enumerate(list_of_contour.get(key)):
            if idx == 0:
                pst_2 = list_of_contour.get(2)[0]
                mssv = draw_rectangle(second, val, pst_2, 160, 100)
            if idx == 1:
                pst_2 = list_of_contour.get(2)[1]
                ma_de = draw_rectangle(second, val, pst_2, 160, 60)

    if key == 2:
        continue
    if key == 6:
        continue

    for idx, val in enumerate(list_of_contour.get(key)):
        # if idx == 0:
        pst_2 = list_of_contour.get(key)[0]
        block = draw_rectangle(second, val, pst_2, 160, 70)
        blocks.append(block)

mssv = cv2.cvtColor(mssv, cv2.COLOR_BGR2GRAY)
mssv_thresh = cv2.threshold(mssv, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# for idx, b in enumerate(blocks):
#     cv2.imshow(f'block: {idx}', b)
#     cv2.imwrite(f'block{idx}.jpg', b)
# cv2.imshow('second', second)
# cv2.imshow('mssv', mssv)
# cv2.imwrite(f'mssv.jpg', mssv)
# cv2.imshow('mssv', mssv_thresh)
# cv2.imwrite(f'mssv_thresh.jpg', mssv_thresh)
#
# cv2.imshow('ma de', ma_de)
# cv2.imwrite(f'ma_de.jpg', ma_de)
#
# cv2.imwrite('col_num.jpg', second)
#
# plt.show()
cv2.imshow('one', warped)
cv2.imshow('second', second)

cv2.waitKey(0)
