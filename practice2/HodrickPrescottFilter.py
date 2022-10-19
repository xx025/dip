import numpy


class HPF:

    def __init__(self, img, mould):
        self.img = img
        self.mould = numpy.array(mould)
        self.img_shape = self.img.shape
        self.mould_shape = self.mould.shape

    def filter(self):
        """
        进行滤波操作
        :return:
        """
        _img_ = numpy.zeros(self.img_shape, dtype='uint8')

        r = self.mould_shape[0] // 2
        for i in range(self.img_shape[0] - self.mould_shape[0]):
            for j in range(self.img_shape[1] - self.mould_shape[1]):
                m1 = self.img[i:i + self.mould_shape[0], j: j + self.mould_shape[1]]
                _img_[i + r][j + r] = numpy.sum(m1 * self.mould)

        return _img_
