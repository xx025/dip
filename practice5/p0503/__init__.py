def fft_en(img):
    """
     频域内对图像进行增强
     https://blog.csdn.net/anqijiayou/article/details/79835853

    :param img:
    :return:
    """

    import numpy as np
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    s1 = np.abs(fshift)
    s1_angle = np.angle(fshift)
    s1_real = s1 * np.cos(s1_angle)
    s1_imag = s1 * np.sin(s1_angle)
    s2 = np.zeros(img.shape, dtype=complex)
    s2.real = np.array(s1_real)
    s2.imag = np.array(s1_imag)
    f2shift = np.fft.ifftshift(s2)
    img_back = np.fft.ifft2(f2shift)
    img_back = np.abs(img_back)
    img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
    return img_back
