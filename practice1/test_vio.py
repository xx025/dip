import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_frames_diff(self):
        import cv2

        """
        测试帧差法
        """

        from practice1.vio_ import Video
        from practice1.vio_ import FrameDiff

        vid = Video(path='../opencv-samples-data/vtest.avi')

        print(len(vid.frames))


        l1 = vid.frames[106]
        l2 = vid.frames[107]

        op = FrameDiff(l1=l1, l2=l2)

        lt = op.frame_diff()
        rel = op.rec_angl(diff=lt)

        cv2.imshow('diff', rel)

        cv2.waitKey()

    def test_bs(self):
        """
        测试背景拆分- 背景生成
        :return:
        """

        from practice1.vio_ import BackSub
        from practice1.vio_ import Video

        vid = Video(path='../practice1/video/Laboratory_raw.avi')

        vid_bs = BackSub(vid=vid)

        bgk = vid_bs.build_back_ground()

        # Pickle any object to file
        D = bgk

        F = open('datafile.pkl', 'wb')
        import pickle
        pickle.dump(D, F)
        F.close()


if __name__ == '__main__':
    unittest.main()
