import cv2
import numpy
from numpy import median


class FilterMould:
    m1 = [[1, 1, 0, 1, 1],
          [1, 1, 0, 1, 1],
          [0, 0, 0, 0, 0],
          [1, 1, 0, 1, 1],
          [1, 1, 0, 1, 1]]

    m2 = [[0, 1, 1, 1, 0],
          [1, 0, 1, 0, 1],
          [1, 1, 0, 1, 1],
          [1, 0, 1, 0, 1],
          [0, 1, 1, 1, 0]]

    m3 = [[1, 0, 1, 0, 1],
          [0, 1, 1, 1, 0],
          [1, 1, 0, 1, 1],
          [0, 1, 1, 1, 0],
          [1, 1, 1, 1, 1]]


class FilterMethod:
    max_filter = 'max'
    min_filter = 'min'
    center_filter = 'center'
    medium_filter = 'medium'


class ImgFilter:

    def __init__(self, img: numpy.ndarray, mould, method: str):
        """

        :param img: 原始图像
        :param mould: 滤波模板
        :param method: 滤波方式
        """

        self.img = img
        self.method = method
        self.mould = numpy.array(mould)
        self.shape_img = self.img.shape
        self.Mshape = self.mould.shape
        self.img_u = self.filter()

    def filter(self):
        """
        进行滤波操作
        :return:
        """
        _img_ = numpy.copy(self.img)
        r = (self.Mshape[0]) // 2 + 1
        m = self.Mshape[0]

        def medium_filter():
            for i in range(self.shape_img[0] - m):
                for j in range(self.shape_img[1] - m):
                    m1 = self.img[i:i + m, j: j + m]
                    _img_[i + r][j + r] = median(m1 * self.mould)

        def max_filter():
            for i in range(self.shape_img[0] - m):
                for j in range(self.shape_img[1] - m):
                    m1 = self.img[i:i + m, j: j + m]
                    _img_[i + r][j + r] = numpy.max(m1 * self.mould)

        def min_filter():
            for i in range(self.shape_img[0] - m):
                for j in range(self.shape_img[1] - m):
                    m1 = self.img[i:i + m, j: j + m]
                    _img_[i + r][j + r] = numpy.min(m1 * self.mould)

        def center_filter():
            for i in range(self.shape_img[0] - m):
                for j in range(self.shape_img[1] - m):
                    m1 = self.img[i:i + m, j: j + m]
                    val = numpy.min(m1 * self.mould) + numpy.min(m1 * self.mould)
                    _img_[i + r][j + r] = val // 2

        if self.method == 'max':
            max_filter()
        elif self.method == 'min':
            min_filter()
        elif self.method == 'center':
            center_filter()
        elif self.method == 'medium':
            medium_filter()
        else:
            medium_filter()
        return _img_


path = "../practice2/imgelpa.png"

img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

img2 = ImgFilter(img=img1,
                 mould=FilterMould.m1,
                 method=FilterMethod.medium_filter)

cv2.imshow('img', img1)
cv2.imshow('img2', img2.img_u)

cv2.waitKey()
