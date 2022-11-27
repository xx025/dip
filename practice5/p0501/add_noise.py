import numpy as np


class AddNoise:
    noise_types = ('gauss', 'rayleigh', 'exponent', 'uniform')

    @staticmethod
    def add_noise(noise_type=None):
        """
        Add noise to the input.
        """
        # assert noise_type in AddNoise.noise_types

        if noise_type == 'gauss':
            return AddNoise.gauss
        if noise_type == 'rayleigh':
            return AddNoise.rayleigh
        if noise_type == 'exponent':
            return AddNoise.exponent
        if noise_type == 'uniform':
            return AddNoise.uniform

    @staticmethod
    def __clip_image(mask, image):
        """
        :param mask: 噪声蒙盖
        :param image: 原始图像
        :return:
        """
        noisy_img = image + mask
        return np.clip(noisy_img, a_min=0, a_max=255)

    @staticmethod
    def gauss(img, mean=0, sigma=25):
        """
        给图片添加高斯噪声
        根据均值和标准差生成符合高斯分布的噪声

        :param img: 图像
        :param mean: 高斯噪声均值
        :param sigma: 高斯噪声标准差
        :return:
        """
        gauss = np.random.normal(mean, sigma, size=img.shape)
        return AddNoise.__clip_image(gauss, img)

    @staticmethod
    def rayleigh(img, a=30.0):
        """
        给图像添加瑞利(Rayleigh)噪声

        :param img: 图像
        :param a: 瑞利噪声偏移量参数
        :return: 添加噪声后的图像
        """
        noise_rayleigh = np.random.rayleigh(a, size=img.shape)
        return AddNoise.__clip_image(noise_rayleigh, img)

    @staticmethod
    def exponent(img, a=10.0):
        """
        给图像添加指数噪声

        :param img: 图像
        :param a: 均值
        :return:
        """
        noise_exponent = np.random.exponential(scale=a, size=img.shape)
        return AddNoise.__clip_image(noise_exponent, img)

    @staticmethod
    def uniform(img, mean=10, sigma=100):
        """
        给图像添加均值噪声

        :param img: 图像
        :param mean: 均值
        :param sigma: 极差
        :return:
        """

        a = 2 * mean - np.sqrt(12 * sigma)
        b = 2 * mean + np.sqrt(12 * sigma)
        noise_uniform = np.random.uniform(a, b, img.shape)
        return AddNoise.__clip_image(noise_uniform, img)
