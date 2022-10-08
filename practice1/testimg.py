import cv2

from practice1.pic_ import IMGU

img = cv2.imread(r'E:\program\txsb\practice1\imgs\img21.png', cv2.IMREAD_GRAYSCALE)

a4 = IMGU.zft(img,draw=True)

print(a4)
# cv2.imshow('res', a4)
#
# cv2.waitKey()
