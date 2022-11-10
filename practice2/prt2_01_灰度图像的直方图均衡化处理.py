"""

灰度图像的直方图均衡化处理；
"""

from copy import deepcopy

import cv2
import matplotlib
from matplotlib import pyplot as plt
from numpy import histogram

matplotlib.use('Qt5Agg')


class ImgHiramEqa:
    def __init__(self, img):
        self.img = img
        self.fx = self.crete_fx()

    def crete_fx(self):
        points = self.img.ravel()
        l1 = histogram(points, bins=[i for i in range(257)])[0]
        l2 = [None for _ in range(len(l1))]
        for i1, j1 in enumerate(l1):
            l2[i1] = sum(l1[:i1 + 1]) / len(points)
        fx = {}

        # 建立映射关系
        for index, j in enumerate(l2):
            fx[str(index)] = int(j * 255 + 0.5)

        return fx

    def cort_img(self):
        img2 = deepcopy(self.img)

        # 进行像素映射处理
        for key, val in self.fx.items():
            img2[img == int(key)] = val
        return img2


path = "../practice2/leuvenA_s.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

img2 = ImgHiramEqa(img=img).cort_img()
plt.hist(img.ravel(), 256)
plt.hist(img2.ravel(), 256)
cv2.imshow('img', img)
cv2.imshow('img2', img2)

plt.show()
