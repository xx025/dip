import cv2 as cv

from practice5.p0501.add_noise import AddNoise
from practice5.p0501.snr import SNR
from pylot_imgshow import img_show

img = cv.imread(r'data/l.jpg', cv.IMREAD_GRAYSCALE)

imgs = [("Original image", img)]
for i in AddNoise.noise_types:
    add_noise = AddNoise.add_noise(i)
    noise_img = add_noise(img)
    a = SNR.snr(noise_img, img)
    b = SNR.psnr(noise_img, img)
    imgs.append((f"{i} snr:{round(a, 2)} psnr{round(b, 2)}", noise_img))

img_show(imgs, sp="2#3")
