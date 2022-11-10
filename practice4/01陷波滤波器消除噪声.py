import cv2 as cv

from practice4.p0401 import sin_noise, spectrum, butterworthNRFilter, img_filter
from pylot_imgshow import img_show

img = cv.imread(r'p0401/data/lena.jpg', cv.IMREAD_GRAYSCALE)
# 读取图片

noise_img = sin_noise(img)
# 给图片添加周期噪声

dftAmpNorm, df_amp = spectrum(noise_img)
# 频谱图

BRFilter = butterworthNRFilter(noise_img)
# 巴特沃斯陷波带阻滤波器

filter_img = img_filter(noise_img, df_amp, BRFilter)
# 对图片进行滤波

# 显示各种图片
img_show(('Original image', img),
         ('Noise image', noise_img),
         ('DFT spectrum', dftAmpNorm),
         ("BW Filter", BRFilter),
         ("Filter image", filter_img),
         sp="2#3")
