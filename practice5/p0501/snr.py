import math

import numpy as np


class SNR:
    """
    Signal to Noise Ratio

    信噪比
    """

    @staticmethod
    def psnr(target, ref):
        """
        PSNR-峰值信噪比

        :param target: 目标图像
        :param ref: 参考图像
        :return:
        """
        assert target.shape == ref.shape

        diff = ref - target
        rmse = math.sqrt(np.mean(diff ** 2.))
        return abs(20 * math.log10(1.0 / rmse))

    @staticmethod
    def snr(target, ref):
        """
        SNR-信噪比

        :param target: 目标图像
        :param ref: 参考图像
        :return:
        """
        signal_source, signal_source_noise = ref, target



        signal_noise = signal_source - signal_source_noise
        mean_signal_source = np.mean(signal_source)
        signal_source = signal_source - mean_signal_source
        snr = 10 * math.log(np.sum(signal_source ** 2) / np.sum(signal_noise ** 2), 10)
        return abs(snr)

    @staticmethod
    def snrs(target, ref):
        return SNR.snr(target, ref), SNR.psnr(target, ref)
