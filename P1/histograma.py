import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

i = imread('example.jpg', flatten=True)

plt.imshow(i)
plt.show()
