import numpy as np
import matplotlib.pyplot as plt


def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1


size = 100
image = np.zeros((size, size, 3), dtype="uint8")
assert image.shape[0] == image.shape[1]

color1 = [255, 128, 0]
color2 = [0, 128, 255]

for i, m in enumerate(np.linspace(0, 1, image.shape[0])):
    for j, n in enumerate(np.linspace(0, 1, image.shape[0])):
        r = lerp(color1[0], color2[0], (n + m) / 2)
        g = lerp(color1[1], color2[1], (n + m) / 2)
        b = lerp(color1[2], color2[2], (n + m) / 2)
        image[i, j, ::-1] = [r, g, b]

plt.figure(1)
plt.imshow(image)
plt.show()