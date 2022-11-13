print("""
参考学习：
    【opencv-python】同态滤波  https://zhuanlan.zhihu.com/p/515812634   
    
""")

import cv2
import numpy as np


def homomorphic_filter(src, d0=2, r1=1.0, rh=2.0, c=4, h=2.0, l=0.5):
    # 图像灰度化处理
    gray = src.copy()
    if len(src.shape) > 2:  # 维度>2
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 图像格式处理
    gray = np.float64(gray)
    print(gray.dtype)
    print(gray.max(), gray.min())

    # 对数域
    gray = np.log(gray + 1.0)
    gray = gray / np.log(256)

    # 归一化
    print(gray.max(), gray.min())

    # 傅里叶变换
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)

    # arange函数用于创建等差数组
    rows, cols = gray.shape
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2),
                       np.arange(-rows // 2, rows // 2))  # 注意，//就是除法

    # 频率域滤波
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1  # filter
    dst_fftshift = Z * gray_fftshift
    # dst_fftshift = (h - l) * dst_fftshift + l

    # 傅里叶反变换（之前是正变换，现在该反变换变回去了）
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)

    # 选取元素的模
    dst = np.abs(dst_ifft)
    print(dst.min(), dst.max())

    # 对数反变换
    dst = np.exp(dst) - 1
    print(dst.min(), dst.max())
    dst = (dst - dst.min()) / (dst.max() - dst.min())  # 归一化
    print(dst.min(), dst.max())
    dst *= 255
    print(dst.min(), dst.max())

    # dst中，比0小的都会变成0，比255大的都变成255
    # uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst
