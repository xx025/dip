import cv2
import numpy as np

print("""

参考学习：

陷波滤波器——周期性降噪    https://blog.csdn.net/jaxonchan/article/details/118913575
MATLAB--数字图像处理 添加周期噪声   https://blog.csdn.net/weixin_44225182/article/details/102484553
python图片傅立叶频谱图分析    https://blog.csdn.net/seasermy/article/details/102736863
陷波带阻滤波器消除周期噪声干扰    https://blog.csdn.net/youcans/article/details/122839594

""")


def sin_noise(img_grey, A=40, u0=50, v0=50):
    """
    添加周期噪声

    :param img_grey: 灰度图像
    :param  A: 振幅
    :param u0: u0,v0关于x轴和y轴确定正弦频率
    :param v0:
    :return:
    """
    c = np.zeros(img_grey.shape, dtype=img_grey.dtype)
    # 生成同型0矩阵

    for i in range(img_grey.shape[0]):
        for j in range(img_grey.shape[1]):
            c[i, j] = A * np.sin(v0 * i + u0 * j)

    return c + img_grey


def spectrum(img):
    """
    绘制傅里叶频谱图

    :param img: 灰度图像
    :return:
    """

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    dft_amp = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    dft_amp_log = np.log(1.0 + dft_amp)
    # 幅度谱对数变换，以便于显示
    dft_amp_norm = np.uint8(cv2.normalize(dft_amp_log, None, 0, 255, cv2.NORM_MINMAX))
    # 归一化为 [0,255]

    return dft_amp_norm, dft_amp


def butterworthNRFilter(img, radius=15, uk=25, vk=16, n=3):
    """
    巴特沃斯陷波带阻滤波器
    https://blog.csdn.net/youcans/article/details/122839594

    :param img:
    :param radius:
    :param uk:
    :param vk:
    :param n:
    :return:
    """

    M, N = img.shape[1], img.shape[0]
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    Dm = np.sqrt((u - M // 2 - uk) ** 2 + (v - N // 2 - vk) ** 2)
    Dp = np.sqrt((u - M // 2 + uk) ** 2 + (v - N // 2 + vk) ** 2)
    D0 = radius
    n2 = 2 * n
    kernel = (1 / (1 + (D0 / (Dm + 1e-6)) ** n2)) * (1 / (1 + (D0 / (Dp + 1e-6)) ** n2))
    return kernel


def img_filter(img, dft_amp, BRFilter):
    """
     陷波带阻滤波器消除周期噪声干扰
    https://blog.csdn.net/youcans/article/details/122839594


    :param img:
    :param dft_amp:
    :param BRFilter:
    :return:
    """

    img_float32 = np.float32(img)  # 将图像转换成 float32
    rows, cols = img.shape  # 图片的高度和宽度

    # (2) 中心化, centralized 2d array f(x,y) * (-1)^(x+y)
    mask = np.ones(img.shape)
    mask[1::2, ::2] = -1
    mask[::2, 1::2] = -1
    f_image = img_float32 * mask  # f(x,y) * (-1)^(x+y)

    # (3) 快速傅里叶变换
    r_padded = cv2.getOptimalDFTSize(rows)  # 最优 DFT 扩充尺寸
    c_padded = cv2.getOptimalDFTSize(cols)  # 用于快速傅里叶变换
    dft_image = np.zeros((r_padded, c_padded, 2), np.float32)  # 对原始图像进行边缘扩充
    dft_image[:rows, :cols, 0] = f_image  # 边缘扩充，下侧和右侧补0
    cv2.dft(dft_image, dft_image, cv2.DFT_COMPLEX_OUTPUT)  # 快速傅里叶变换

    # (5) 在频率域修改傅里叶变换: 傅里叶变换 点乘 陷波带阻滤波器
    dft_filter = np.zeros(dft_image.shape, dft_image.dtype)  # 快速傅里叶变换的尺寸(优化尺寸)
    for i in range(2):
        dft_filter[:r_padded, :c_padded, i] = dft_image[:r_padded, :c_padded, i] * BRFilter

    # (6) 对频域滤波傅里叶变换 执行傅里叶逆变换，并只取实部
    idft = np.zeros(dft_amp.shape, np.float32)  # 快速傅里叶变换的尺寸(优化尺寸)
    cv2.dft(dft_filter, idft, cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE + cv2.DFT_SCALE)

    # (7) 中心化, centralized 2d array g(x,y) * (-1)^(x+y)
    mask2 = np.ones(dft_amp.shape)
    mask2[1::2, ::2] = -1
    mask2[::2, 1::2] = -1
    idft_cen = idft * mask2  # g(x,y) * (-1)^(x+y)

    # (8) 截取左上角，大小和输入图像相等
    idft_cen_clip = np.clip(idft_cen, 0, 255)  # 截断函数，将数值限制在 [0,255]
    img_filtered = idft_cen_clip.astype(np.uint8)
    img_filtered = img_filtered[:rows, :cols]

    return img_filtered
