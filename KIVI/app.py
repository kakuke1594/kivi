import cv2
import math
import numpy as np
from scipy.spatial import distance as dist
from collections import defaultdict


HORIZONTAL = 0
VERTICAL = 1

MSSV = 'MSSV'
MA_DE = 'MADE'
BLOCKS = 'BLOCKS'


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


def detect_core_components(warped):
    gray = cv2.cvtColor(warped.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    img, contour, hierrachy = cv2.findContours(thresh, 1, 2)
    boundary = list()
    for idx, cnt in enumerate(contour):
        area = cv2.contourArea(cnt)
        if 70 < area < 500:
            # cv2.drawContours(warped, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # black_point.append(approx.reshape(4, 2))
            x, y, w, h = cv2.boundingRect(approx)
            boundary.append([x, y, w, h])
            cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(boundary) != 4:
        print('Error when take picture')

    four_point = sort_pts(boundary)
    four_point = np.array(four_point)
    core_component_warped = four_point_transform(warped, four_point)
    return core_component_warped


def sort_pts(raw_pts):
    # sort base on x-axis
    raw_points = sorted(raw_pts, key=lambda x: x[0])
    for idx, pts in enumerate(raw_points):
        try:
            if abs(pts[0]-raw_points[idx+1][0]) < 5:
                raw_points[idx+1][0] = pts[0]
        except IndexError:
            print('pass index')
    # Get  tl, tr, bl, br
    raw_points = sorted(raw_points, key=lambda x: (x[0], x[1]))

    array_point = list()
    for idx, pts in enumerate(raw_points, start=1):
        new_pts = get_order_point(pts, idx)
        array_point.append(new_pts)
    return array_point


def get_order_point(pts, idx):
    (x, y, w, h) = pts
    if idx == 1:
        return [x, y]
    elif idx == 2:
        return [x, y + w]
    elif idx == 3:
        return [x + h, y]
    elif idx == 4:
        return [x+w, y+h]


def detect_small_components(image):
    mini_square = list()
    mini_bouding = list()
    mini_cont = list()

    mask = cv2.inRange(image, (0, 0, 0, 0), (180, 255, 90, 0))
    # cv2.imshow('a', image)
    _, cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for ca in cont:
        area = cv2.contourArea(ca)
        if 41 < area < 100:
            cv2.drawContours(image, ca, -1, (255, 0, 255), 1)
            peri = cv2.arcLength(ca, True)
            approx = cv2.approxPolyDP(ca, 0.02 * peri, True)
            mini_square.append(approx)
            mini_cont.append(ca)
            x, y, w, h = cv2.boundingRect(approx)
            mini_bouding.append((x, y, w, h))

    cv2.imshow('çont', image)
    mini_cont.sort(key=lambda x: get_contour_precedence(x, image.shape[1]))

    # sort left to right, top to bottom
    new_pos = dict()
    cols = defaultdict(list)
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
            elif y_value - 5 < y < y_value + 5:
                cols[col].append((x, y, w, h))
            else:
                col += 1
                cols[col].append((x, y, w, h))
                y_value = y

    list_of_contour = defaultdict(list)
    for column in cols:
        for c in cols[column]:
            # image = cv2.putText(image, str(column), cv2.boundingRect(new_pos.get(c))[:2], cv2.FONT_ITALIC, 1, [125])
            list_of_contour[column].append(new_pos.get(c))
    cv2.imshow('second', image)
    return list_of_contour


def crop_block(contours, image):
    blocks = list()
    mssv = None
    ma_de = None

    for key in list(contours):
        not_block = False
        if key == 1:
            for idx, val in enumerate(contours.get(key)):
                if idx == 0:
                    bot_points = contours.get(2)[0]
                    raw_height = int(bot_points.tolist()[0][0][1] - val.tolist()[0][0][1])
                    h = roundup(raw_height)
                    mssv = draw_rectangle(image, val, bot_points, h, 110)
                    not_block = True
                    continue
                if idx == 1:
                    bot_points = contours.get(2)[1]
                    raw_height = int(bot_points.tolist()[0][0][1] - val.tolist()[0][0][1])
                    h = roundup(raw_height)
                    ma_de = draw_rectangle(image, val, bot_points, h, 80)
                    not_block = True
                    continue

        if key == 2:
            continue
        elif key == 6:
            continue
        elif not_block:
            continue

        sub_block = list()
        for idx, val in enumerate(contours.get(key)):
            bot_points = contours.get(key)[0]
            raw_height = int(bot_points.tolist()[0][0][1] - val.tolist()[0][0][1])
            # h = roundup(raw_height)
            block = draw_rectangle(image, val, bot_points, 170, 80)
            sub_block.append(block)
        blocks.append(sub_block)
    return mssv, ma_de, blocks


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def draw_rectangle(img, pts_1, pst_2, h, w):
    x = pts_1.tolist()[0]

    crop_img = img[x[0][1]: x[0][1] + h, x[0][0]: x[0][0] + w]
    return crop_img


def detect_multiple_choice_quiz(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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

    multiple_sheet = four_point_transform(image, displayCnt.reshape(4, 2))
    return multiple_sheet


def get_display_cnts(cnts):
    displayCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            displayCnt = approx
            break
    return displayCnt


def read_image():
    image = cv2.imread("img/IMG_6011.JPG")
    copy_image = image.copy()
    return image, copy_image


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


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


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


def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
    # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
    for x, y in zip(x_[x], y_[y]):
        yield x, y


def find_circle(block, dp=None, is_vertical=False, p2=40):
    if dp is None:
        print("Require dp when find circle")
    gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT, dp=dp,
                               minDist=10, param1=200, param2=35,
                               minRadius=1, maxRadius=10)

    # circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT, dp=7,
    #                            minDist=10, param1=200, param2=40,
    #                            minRadius=1, maxRadius=10)

    circles = np.round(circles[0, :]).astype("int")
    if is_vertical:
        circles = sorted(circles, key=lambda v: [v[0], v[1]])
        num_question = 10
    else:
        circles = sorted(circles, key=lambda v: [v[1], v[0]])
        num_question = 4

    # round circle to int
    circles = np.uint16(np.around(circles))
    return circles, num_question


def grade_block(image, circles, batch, type=None):
    answer = dict()
    a = 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mssv_thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    results = list()
    for (q, i) in enumerate(np.arange(0, len(circles), batch)):
        bubbled = None
        ctns = circles[i: i + batch]
        # print(f'{i}-{i+batch}')
        for (inner_idx, circle) in enumerate(ctns):
            (x, y, r) = circle
            rectX = (x - r)
            rectY = (y - r)
            crop_img = mssv_thresh[rectY:(rectY + 3 * r), rectX:(rectX + 3 * r)]
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)
            total = cv2.countNonZero(crop_img)

            # TODO: add filter if all bubble in each question is small than 70
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, i + inner_idx, (x, y, r))

        cv2.circle(image, radius=bubbled[2][2], center=(bubbled[2][0], bubbled[2][1]), color=(0, 255, 0), thickness=-1)
        results.append(bubbled)

    answer = ''.join(str(bubble[1] % batch) for bubble in results)
    return image, answer


def find_marked_bubble(mssv, ma_de, blocks):
    # ma so sinh vien
    mssv_circles, batch = find_circle(mssv, dp=7.3, is_vertical=True)
    result, mssv_answer = grade_block(mssv, mssv_circles, batch)
    print('ma so sinh vien: ', mssv_answer)
    # ma de
    ma_de_circles, batch = find_circle(ma_de, dp=7, is_vertical=True)
    result, ma_de_answer = grade_block(ma_de, ma_de_circles, batch)
    print('ma de: ', ma_de_answer)

    # blocks
    if len(blocks) == 3:
        # sort answer block from top to bot, then left to right
        blocks = [list(a) for a in zip(blocks[0], blocks[1], blocks[2])]

    question_ans = dict()
    i = 0
    for idx, sub_blocks in enumerate(blocks, start=1):
        for idx_sub, sub_block in enumerate(sub_blocks):
            # test = cv2.putText(sub_block, str(i), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            block_circles, batch = find_circle(sub_block, 7.5, p2=35)
            # if len(block_circles) != 40:
                # improve quality and retry
                # print(f'block {i}: have {len(block_circles)}')
            image, answer = grade_block(sub_block, block_circles, batch)
            # cv2.imshow(f'block{i}', image)
            i += 1
            question_ans[i] = answer

    print(question_ans)
    return mssv_answer, ma_de_answer, question_ans


def main():
    image = cv2.imread("img/IMG_6011.JPG")
    # resize for easily to view
    image_resized = image_resize(image, 800, 800)
    # TODO: add rotation if photo is not portrait
    multile_choice_quiz = detect_multiple_choice_quiz(image_resized)
    core_component = detect_core_components(multile_choice_quiz)
    small_components_contour = detect_small_components(core_component)
    # TODO: add filter if components is not enough
    mssv, ma_de, blocks = crop_block(small_components_contour, core_component)

    mssv, ma_de, cau_tra_loi = find_marked_bubble(mssv, ma_de, blocks)
    # grade_mutiple_choice_quiz(ma_de, cau_tra_loi)
    # save_the result =

    cv2.waitKey(0)



if __name__ == '__main__':
    main()
