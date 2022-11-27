import cv2 as cv

from practice5.p0501.add_noise import AddNoise
from practice5.p0501.snr import SNR
from practice5.p0503 import fft_en

img = cv.imread(r'data/lena.jpg', cv.IMREAD_GRAYSCALE)

for i in AddNoise.noise_types:
    add_noise = AddNoise.add_noise(i)
    noise_img = add_noise(img)

    a1, b1 = SNR.snrs(noise_img, img)

    ls_img = cv.GaussianBlur(noise_img, (5, 5), 0)

    a2, b2 = SNR.snrs(ls_img, img)

    fft_img = fft_en(noise_img)

    a3, b3 = SNR.snrs(fft_img, img)

    print("{}-noise_img:"
          " \t snr:{:.2f} \t psnr{:.2f} "
          " \n \t -{} \t snr:{:.2f} \t  psnr{:.2f}"
          " \n \t -{} \t snr:{:.2f} \t  psnr{:.2f}"
          .format(add_noise.__name__, a1, b1,
                  cv.GaussianBlur.__name__, a2, b2,
                  fft_en.__name__, a3, b3))
