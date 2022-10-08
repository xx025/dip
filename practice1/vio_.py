import cv2


class Video:

    def __init__(self, path):
        self.video = cv2.VideoCapture(path)  # 视频对象
        self.frames = self.__all_frames()  # 所有帧
        self.height, self.width, self.RGB3 = self.frames[0].shape  # 高度和宽度
        self.lens = len(self.frames)  # 帧数

    def __all_frames(self):
        frames = []
        capture = self.video
        while True:
            ret, img = capture.read()  # img 就是一帧图片
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            frames.append(img)
            if not ret:
                break  # 当获取完最后一帧就结束
        return frames


def frame_pre_op(ll):
    ll = cv2.cvtColor(ll, cv2.COLOR_BGR2GRAY)
    # 彩色转灰度

    ll = cv2.GaussianBlur(ll, (3, 3), 1)
    # 高斯滤波

    return ll


class FrameDiff:

    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

    def frame_diff(self):
        """
        Frame difference 帧差法

        两帧图像计算插值寻找边缘

        :return: 两个图像的差值
        """

        l1 = FrameDiff.__pre_tr(self.l1)
        l2 = FrameDiff.__pre_tr(self.l2)
        # 预处理

        diff = cv2.absdiff(l1, l2)
        diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理

        es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
        diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀

        # https://blog.csdn.net/drippingstone/article/details/116081434
        # code line: 33-36

        return diff

    @staticmethod
    def __pre_tr(ll):
        # 对帧进行预处理，先转灰度图，再进行高斯滤波。
        # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、
        # 光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运
        # 动和跟踪时将其检测出来。
        # https: // blog.csdn.net / qq_45832961 / article / details / 122351534
        return frame_pre_op(ll)

    def rec_angl(self, diff):
        # 绘制矩形框
        # https://blog.csdn.net/drippingstone/article/details/116081434
        # code line: 33-36
        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
        for c in contours:
            if cv2.contourArea(c) < 1500:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
                continue
            (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
            cv2.rectangle(self.l2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return self.l2


class BackSub:
    """
    背景拆分法
    """

    def __init__(self, vid: Video):
        """

        :param vid: 一个Video 对象
        """
        self.width = vid.width
        self.height = vid.height
        self.frames = vid.frames
        self.lens = vid.lens
        pass

    def build_back_ground(self):
        """
        构建一个背景，统计所有帧在像素点位置出现像素值的次数取众数在进行均值

        注：特别耗时 F(t)=帧数*宽度*高度
        :return: 一个背景图像
        """
        m_ = [[dict() for i in range(self.width)] for j in range(self.height)]

        for _i in range(self.lens):
            print(f"进度[{_i}/{self.lens}]")
            for j in range(self.height):
                for i in range(self.width):
                    sg = frame_pre_op(self.frames[_i])
                    ind = str(sg[j][i])
                    if m_[j][i].get(ind):
                        m_[j][i][ind] += 1
                    else:
                        m_[j][i][ind] = 1
        return m_
