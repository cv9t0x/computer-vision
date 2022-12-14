import cv2
import numpy as np

cap = cv2.VideoCapture("./res/video2.mp4")
cv2.namedWindow("Frames")

while cap.isOpened():
    ret, frame = cap.read()

    if frame is not None:
        img = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    _temp, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=100, param2=50, minRadius=10, maxRadius=25)

    contours = cv2.findContours(thresh, 1, 2)[0]

    rect_c = 0
    circle_c = 0

    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.025*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w) / h
            if ratio <= 0.5 and w < 25 and h > 50:
                cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
                rect_c += 1

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255), 2)
            circle_c += 1

    if (rect_c == 1 and circle_c == 1):
        cv2.imwrite("./out/out.jpg", img)
        break

    if ret:
        cv2.imshow("Frames", frame)

    key = cv2.waitKey(25)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
