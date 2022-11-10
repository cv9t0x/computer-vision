import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu


def get_color(image):
    hue = np.unique(image[:, :, 0]) * 360
    hue = np.max(hue[hue > 0])

    if 0 < hue <= 20 or hue > 330:
        return 'red'
    elif 20 < hue <= 40:
        return 'orange'
    elif 40 < hue <= 75:
        return 'yellow'
    elif 75 < hue <= 165:
        return 'green'
    elif 165 < hue <= 190:
        return 'lime'
    elif 190 < hue <= 275:
        return 'blue'
    elif 275 < hue <= 330:
        return 'purple'


image = plt.imread('./balls_and_rects.png')

threshold = threshold_otsu(color.rgb2gray(image))
binary = image > threshold

hsv_image = color.rgb2hsv(image)

labeled = label(binary)

rects = {}
circles = {}

regions = regionprops(labeled)

for region in regions:
    y_min, x_min, _, y_max, x_max, _ = region.bbox

    coloured = hsv_image[y_min:y_max, x_min:x_max]
    color = get_color(coloured)

    if (np.all(region.image)):
        if color in rects:
            rects[color] += 1
        else:
            rects[color] = 1
    else:
        if color in circles:
            circles[color] += 1
        else:
            circles[color] = 1

print("-----------")

print("Rects:")

for color in rects:
    amount = rects[color]
    print(f"{amount} {color}")

print("-----------")

print("Circles:")

for color in circles:
    amount = circles[color]
    print(f"{amount} {color}")

print("-----------")

print(f"Total: {labeled.max()}")

print("-----------")
