# P68

import cv2
import numpy


class Laplace:
    m1 = [[0, -1, 0],
          [-1, 4, -1],
          [0, -1, 0]]
    m2 = [[-1, -1, -1],
          [-1, 8, -1],
          [-1, -1, -1]]


class ImgSharpen:

    def __init__(self, img: numpy.ndarray, mould: list):
        self.img = img
        self.mould = numpy.array(mould)
        self.Ishape = self.img.shape
        self.Mshape = self.mould.shape

        self.img_outline = self.filter()
        self.fil_tran_img = self.translation()

    def filter(self):
        """
        进行滤波操作
        :return:
        """
        _img_ = numpy.zeros(self.Ishape, dtype='uint8')

        for i in range(self.Ishape[0] - self.Mshape[0]):
            for j in range(self.Ishape[1] - self.Mshape[1]):
                m1 = self.img[i:i + self.Mshape[0], j: j + self.Mshape[1]]
                val = numpy.sum(m1 * self.mould)
                _img_[i + 1][j + 1] = 0 if val < 0 else val

        return _img_

    def translation(self):
        """
        进行灰度平移

        :return:
        """
        t_img = self.img + self.img_outline
        return numpy.array(t_img, dtype='uint8')


path = "../opencv-samples-data/lena.jpg"

img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

img2 = ImgSharpen(img=img1, mould=Laplace.m1)

cv2.imshow('img', img1)
cv2.imshow('img2', img2.img_outline)
cv2.imshow('img3', img2.fil_tran_img)

cv2.waitKey()
