import cv2
import numpy as np
import random

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cam.set(cv2.CAP_PROP_EXPOSURE, -4)


def get_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def get_mask_contours(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


colors = {
    "yellow": {
        "lower": np.array([20, 140, 120]),
        "upper": np.array([60, 255, 255])
    },
    "green": {
        "lower": np.array([60, 140, 50]),
        "upper": np.array([80, 255, 255]),
    },
    "blue": {
        "lower": np.array([90, 200, 50]),
        "upper": np.array([105, 255, 255])
    }
}

sequency = random.sample(["green", "yellow", "blue"], 3)

print("Sequency: " + ", ".join(sequency))

flag = False

while cam.isOpened():
    ret, frame = cam.read()

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    masks = []
    mask_contours = []
    coords = {}

    for color in colors:
        variants = colors[color]
        mask = get_mask(hsv, variants["lower"], variants["upper"])
        contours = get_mask_contours(mask)
        masks.append(mask)
        mask_contours.append(contours)

    for index, contours in enumerate(mask_contours):
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(
                    radius), (0, 255, 0), 2)
                color = list(colors.keys())[index]
                coords[color] = x

    sorted_coords = sorted(coords.items(), key=lambda item: item[1])
    res = map(lambda coord: coord[0], sorted_coords)

    if (list(res) == sequency and not flag):
        print("Win!")
        flag = True
    else:
        flag = False

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    cv2.imshow("Camera", frame)

cam.release()
cv2.destroyAllWindows()
