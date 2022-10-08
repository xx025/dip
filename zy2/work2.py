'''
编码实现图像的缩放、平移、裁剪、旋转等处理。

'''

import cv2 as cv
import numpy as np
from cv2 import ROTATE_180

img = cv.imread('imgs/image.png')

height, width = img.shape[:2]

# 缩放
res_zoom = cv.resize(src=img, dsize=(int(0.5 * width), int(0.5 * height)))
# 缩小0.5倍

# 平移
mat_translation = np.float32([[1, 0, -20], [0, 1, -50]])
# 变换矩阵：设置平移变换所需的计算矩阵：2行3列
# 平移 x +20 y -50
res_shift = cv.warpAffine(img, mat_translation, (width - 20, height - 50))

# 裁剪
res_clip = img[0:200, 0:200]
# 裁剪坐标为[y0:y1, x0:x1]

# 旋转
res_rotate = cv.rotate(src=img, rotateCode=ROTATE_180)
# 原图旋转180度

cv.imshow('res', img)
cv.imshow('zoom', res_zoom)
cv.imshow('shift', res_shift)
cv.imshow('clip', res_clip)
cv.imshow('rotate', res_rotate)

cv.waitKey()
