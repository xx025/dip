import unittest


class MyTestCase(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_f1(self):
        """
        灰度图像的直方图计算和显示
        :return:
        """
        import cv2
        from practice1.pic_ import IMGU

        img = cv2.imread('../opencv-samples-data/left01.jpg', cv2.IMREAD_GRAYSCALE)

        IMGU.zft(img, draw=True)
        cv2.imshow('img', img)
        cv2.waitKey()

    def test_f2(self):
        # 灰度图像的负像，
        import cv2
        from practice1.pic_ import IMGU

        img = cv2.imread('../opencv-samples-data/left01.jpg', cv2.IMREAD_GRAYSCALE)
        a4 = IMGU.fx(img)
        cv2.imshow('res', a4)
        cv2.waitKey()

    def test_f3(self):
        # 灰度级转换

        import cv2
        from practice1.pic_ import IMGU

        img = cv2.imread('../opencv-samples-data/left01.jpg', cv2.IMREAD_GRAYSCALE)
        a4 = IMGU.covert_to_16(img)
        cv2.imshow('res', a4)
        cv2.waitKey()

    def test_f4(self):
        # 生成噪声图
        import cv2
        from practice1.pic_ import IMGU

        img = cv2.imread('../opencv-samples-data/left01.jpg', cv2.IMREAD_GRAYSCALE)
        a4 = IMGU.noise(img, sigma=20)
        cv2.imshow('res', a4[1])
        cv2.waitKey()

    def test_f5(self):
        # 图像相加去噪
        import cv2
        from practice1.pic_ import IMGU

        img = cv2.imread('../opencv-samples-data/left01.jpg', cv2.IMREAD_GRAYSCALE)
        imgs = IMGU.noise(img, sigma=20)
        img = IMGU.mean(imgs)
        cv2.imshow('res', img)
        cv2.waitKey()


if __name__ == '__main__':
    unittest.main()
