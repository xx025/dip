# 来源： https://opencv.apachecn.org/#/docs/4.0.0/6.3-tutorial_py_bg_subtraction?id=backgroundsubtractormog2
import cv2 as cv

cap = cv.VideoCapture('../opencv-4.6.0/samples/data/vtest.avi')

fgbg = cv.createBackgroundSubtractorMOG2()

while (1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv.imshow('frame', fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
