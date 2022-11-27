import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from practice5.p0501.add_noise import AddNoise

img_ori = cv.imread(r'data/l.jpg', cv.IMREAD_GRAYSCALE)

img_gauss = AddNoise.gauss(img=img_ori, mean=50, sigma=20)
img_rayleigh = AddNoise.rayleigh(img=img_ori, a=20)
img_exponent = AddNoise.exponent(img=img_ori, a=20)
img_uniform = AddNoise.uniform(img=img_ori, mean=20, sigma=50)

bins = np.linspace(0, 255, 256)
x = np.arange(1, 100)
plt.figure(figsize=(12, 8), dpi=200)
plt.subplot(231)
plt.imshow(img_ori, cmap='gray')
plt.subplot(232)
plt.hist(img_ori.flatten(), bins, density=True, alpha=0.7)
plt.title('img_ori')
plt.subplot(233)
plt.hist((img_gauss - img_ori).flatten(), bins, density=True, alpha=0.7)
plt.title('gauss')
plt.subplot(234)
plt.hist((img_exponent - img_ori).flatten(), bins, density=True, alpha=0.7)
plt.title('exponent')
plt.subplot(235)
plt.hist((img_rayleigh - img_ori).flatten(), bins, density=True, alpha=0.7)
plt.title('rayleigh')
plt.subplot(236)
plt.hist((img_uniform - img_ori).flatten(), bins, density=True, alpha=0.7)
plt.title('uniform')
plt.show()
