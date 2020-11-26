import cv2
import numpy as np
from collections import defaultdict


def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
    # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
    for x, y in zip(x_[x], y_[y]):
        yield x, y


def points_mode_value(img, points):
    values = [img[y, x] for y, x in points]
    mode = max(set(values), key=values.count)
    return mode


def mode(img, bubbles):
    small = 300
    obj = None
    for bubble in bubbles:
        points = points_in_circle_np(x0=bubble[1], y0=bubble[0], radius=bubble[2])
        mode = points_mode_value(img, points)
        if mode < small and mode != 0:
            obj = bubble
            small = mode
    return obj


def main():
    img = cv2.imread('block0.jpg', )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    thresh
    circles = cv2.HoughCircles(
        image=img,
        method=cv2.HOUGH_GRADIENT,
        dp=7.1,
        minDist=10,
        param1=200,
        param2=15,
        minRadius=1,
        maxRadius=10
    )
    print(circles.shape)

    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=lambda v: [v[1], v[0]])
    img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cols = defaultdict(list)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for cont, i in enumerate(circles, start=1):
            cols[i[0]].append(i)

        for col, bubbles in cols.items():
            black_box = mode(img, bubbles)
            cv2.circle(img_colored, (black_box[0], black_box[1]), black_box[2], (0, 0, 255), -1)  # cv2.FILLED)

            #
            # if mode == 255:
            # else:
            #     cv2.circle(img_colored, (i[0], i[1]), i[2], (0, 255, 0), -1)  # cv2.FILLED)
            #
            # cv2.putText(img_colored, str(cont), (i[0] - i[2], i[1] + i[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
            #             cv2.LINE_AA)

    cv2.imwrite("out1.png", img_colored)


if __name__ == "__main__":
    main()
