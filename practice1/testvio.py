import cv2

from practice1.vio_ import Video

vid = Video(path='../practice1/video/Laboratory_raw.avi')

print(len(vid.frames))

# l1 = vid.frames[100]
# l2 = vid.frames[101]

# op = FrameDiff(l1=l1, l2=l2)
#
# lt = op.frame_diff()
# rel = op.rec_angl(diff=lt)

# cv2.imshow('diff', rel)
#
# cv2.waitKey()


fgbg = cv2.createBackgroundSubtractorMOG2()


fgmask = fgbg.apply(vid.frames[1])

# cv2.imshow('11', fgbg)
cv2.imshow('12', fgmask)
for i in vid.frames:
    fgmask = fgbg.apply(i)
    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.waitKey()
# cv2.destroyAllWindows()
