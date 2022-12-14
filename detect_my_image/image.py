import cv2
import numpy as np


img = cv2.imread('./res/image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 5)
ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU)

circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
                           param1=100, param2=50, minRadius=10, maxRadius=25)

contours = cv2.findContours(thresh, 1, 2)[0]

for cnt in contours:
    x1, y1 = cnt[0][0]
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(w) / h
        if ratio <= 0.5 and w < 25 and h > 50:
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
