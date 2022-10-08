import random
from copy import copy

import matplotlib
import numpy as np


class IMGU:

    @staticmethod
    def zft(img, draw=False):
        """
        直方图数据计算 和绘制直方图
        """
        d2 = {}
        for i in range(len(img)):
            for j in range(len(img[0])):
                if d2.get(str(img[i][j])):
                    d2[str(img[i][j])] += 1
                else:
                    d2[str(img[i][j])] = 1
        if draw:
            from matplotlib import pyplot as plt

            matplotlib.use('Qt5Agg')
            # 显示直方图
            plt.hist(img.ravel(), 256)
            plt.show()
        return d2

    @staticmethod
    def covert_to_16(img):
        # 除以15进行四舍五入 取整
        for i in range(len(img)):
            for j in range(len(img[0])):
                img[i][j] = np.around(img[i][j] / 15) * 15
        return img

    @staticmethod
    def fx(img):
        """
        灰度图片进行反相
        """
        return 255 - img

    @staticmethod
    def noise(img, sigma=5, N=10):

        """
        :param img: 原始图像，
        :param sigma: 随机数范围，可理解为强度
        :param N: 产生数据量（图片张数）
        :return: 多张施加噪声后的数据列表
        """
        mu = 0  # 均值0
        re = []
        for _s in range(N):
            _img = copy(img)
            for i in range(len(img)):
                for j in range(len(img[0])):
                    img[i][j] += random.gauss(mu, sigma)
            re.append(_img)
        return re

    @staticmethod
    def mean(imgs):
        """
        :param imgs: 带有噪声的图像集合
        :return: 均值后的图像
        """
        q = np.array(imgs[0], dtype=int)
        # 取第一张图像进行加和
        # 转换数据类型，防止溢出

        for _i in range(1, len(imgs)):
            for i in range(len(q)):
                for j in range(len(q[0])):
                    q[i][j] += imgs[_i][i][j]
        q = q // len(imgs)
        # 求均值（取整）
        img1 = np.array(q, dtype='uint8')
        # 重新转换成 uint8 灰度图像类型
        return img1
