"""
同态滤波增强
"""
import cv2

from practice4.p0402 import homomorphic_filter
from pylot_imgshow import img_show

img = cv2.imread(r'p0402/data/lena_light.jpg', 0)
# 读取图片


filter_img = homomorphic_filter(img, r1=0.2, rh=1.5)
# 将图片执行同态滤波器
imgs = [('Original image', img), ("Filter image", filter_img)]
img_show(imgs, sp="2#1")
