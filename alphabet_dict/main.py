import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from collections import defaultdict


def count_lakes_and_bays(prop, cut=False):
  b = ~prop.image
  if cut:
    b = b[1:-1, 1:-1]
  lb = label(b)
  regs = regionprops(lb)
  count_lakes = 0
  count_bays = 0
  for reg in regs:
    flag = False
    for y, x in reg.coords:
      if (y == 0 or
          x == 0 or
          y == b.shape[0] - 1 or
          x == b.shape[1] - 1):
          flag = True
          break
    if not flag:
      count_lakes += 1
    else:
      count_bays += 1
  return count_lakes, count_bays


def has_vline(prop):
  return 1 in prop.image.mean(0)


def filling_factor(prop):
  return np.sum(prop.image) / prop.image.size


def get_area(prop, label):
  return np.array(np.where(prop == label, prop)).flatten().size


def find_centroid(prop, axis="X", offset="end"):
  cy, cx = prop.image.shape[0] // 2, prop.image.shape[1] // 2
  if axis == "X":
    if offset == "start":
      return prop.image[cy, 0] > 0
    elif offset == "center":
      return prop.image[cy, cx] > 0
    elif offset == "end":
      return prop.image[cy, -1] > 0
  elif axis == "Y":
    if offset == "start":
      return prop.image[0, cx] > 0
    elif offset == "center":
      return prop.image[cy, cx] > 0
    elif offset == "end":
      return prop.image[-1, cx] > 0


def recorgnize(image):
  result = defaultdict(lambda: 0)
  labeled = label(image)
  props = regionprops(labeled)
  for prop in props:
    (lakes, bays) = count_lakes_and_bays(prop)
    if np.all(prop.image):
      result["-"] += 1
    elif lakes == 2:
      if has_vline(prop):
        result["B"] += 1
      else:
        result["8"] += 1
    elif lakes == 1:
      if bays == 3:
        result["A"] += 1
      elif bays == 2:
        if has_vline(prop):
          if find_centroid(prop, axis="Y", offset="center"):
            result["P"] += 1
          else:
            result["D"] += 1
      else:
        result["0"] += 1
    elif lakes == 0:
      if has_vline(prop):
        result["1"] += 1
      elif bays == 2:
        result["/"] += 1
      (lakes, bays) = count_lakes_and_bays(prop, True)
      if bays == 4:
        result["X"] += 1
      elif bays == 5:
        cy, cx = prop.image.shape[0] // 2, prop.image.shape[1] // 2
        if prop.image[cy, cx] != 0:
          result["*"] += 1
        else:
          result["W"] += 1
    else:
      result["unknown"] += 1
  return result


img = plt.imread("./symbols.png")
img = np.mean(img, 2)
img[img > 0] = 1

rec = recorgnize(img)
print(rec)
print(round((1.0 - rec[None] / sum(rec.values())) * 100, 2), '%')