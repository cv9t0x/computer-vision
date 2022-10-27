from matplotlib import pyplot as plt
import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_opening


square_mask = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
c_mask = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])

masks = []

masks.append(square_mask)
masks.append(c_mask)
masks.append(np.rot90(square_mask))
for i in range(3):
  masks.append(np.rot90(c_mask, i + 1))

image = np.load('./ps.npy.txt')
labeled = label(image)

print(f"All: {np.max(labeled)}")
for mask in masks:
  labeled = label(binary_opening(image, mask))
  print(f"{np.max(labeled.ravel())}:\n{mask}", end="\n\n")