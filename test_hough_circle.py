import cv2
import numpy as np
from collections import defaultdict



image = cv2.imread("mssv.jpg")
output = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(gray, (7, 7), cv2.BORDER_DEFAULT)

# detect circles in the image
circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT, dp=7.1,
                           minDist=10, param1=200, param2=15,
                           minRadius=1, maxRadius=10)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    i = 0
    quest = defaultdict(list)
    circles_sorted = sorted(circles, key=lambda v: [v[0], v[1]])
    for data in circles_sorted:
        (x, y, r) = data
        # draw the circle in the output image, then draw a rectangle2
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 1)
        print("Circle ", i)
        i += 1
        quest[x].append(data)
    # show the output image
    # for col, ans in quest.items():
    #     for answer in ans:

    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)

