import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters.thresholding import threshold_otsu
from skimage.measure import regionprops, label
from pathlib import Path

count = 0

for path in Path("./images").glob("*.jpg"):
    img = plt.imread(path)
    img = img[40:-40, 40:-40]
    gray_img = rgb2gray(img)

    thresh = threshold_otsu(gray_img)
    binary = gray_img.copy() <= thresh

    labeled = label(binary)
    regions = regionprops(labeled)

    for region in regions:
        if region.eccentricity > 0.98 and region.equivalent_diameter >= 55.0:
            count += 1

    print(f"Processed {path}")

print(f"Number of pencils: {count}")
