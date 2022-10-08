'''


帧差法依据的原则是：当视频中存在移动物体的时候，相邻帧（或相邻三帧）之间在灰度上会有差别，求取两帧图像灰度差的绝对值，则静止的物体在差值图像上表现出来全是0，
而移动物体特别是移动物体的轮廓处由于存在灰度变化为非0，当绝对值超过一定阈值时，即可判断为运动目标，从而实现目标的检测功能。

 帧间差分法的优点是算法实现简单，程序设计复杂度低；对光线等场景变化不太敏感，能够适应各种动态环境，有着比较强的鲁棒性。
 缺点是不能提取出对象的完整区域，对象内部有“空洞”，只能提取出边界，边界轮廓比较粗，往往比实际物体要大。对快速运动的物体，
 容易出现鬼影的现象，甚至会被检测为两个不同的运动物体，对慢速运动的物体，当物体在前后两帧中几乎完全重叠时，则检测不到物体。
'''

import cv2 as cv


camera = cv.VideoCapture(0)

if (camera.isOpened()):
  print('摄像头已打开')
else:
  print('摄像头未打开')

# 测试使用查看视频size
size = (int(camera.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(camera.get(cv.CAP_PROP_FRAME_HEIGHT)))
print('size:'+repr(size))

# 第二步 构造形态学使用的kernel
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
# 第三部构造高斯混合模型
model = cv.createBackgroundSubtractorMOG2()

# 读取视频中的图片，并使用高斯模型进行拟合
while True:
    ret , frame = camera.read()
    # 运用高斯模型进行拟合，在两个标准差内设置为 0 ，在两个标准差外设置为255
    fgmk = model.apply(frame)
    # 第五步，使用形态学的开运算做背景的去除
    fgmk = cv.morphologyEx(fgmk,cv.MORPH_OPEN,kernel)

    # 第六步 计算fgmk的轮廓
    contours = cv.findContours(fgmk.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[1]# 计算一幅图像中目标的轮廓

    for c in contours:
        if cv.contourArea(c) < 1500:
            continue
        (x,y,w,h)=cv.boundingRect(c) # 计算矩形边界框

        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # 图片展示
    cv.imshow('fgmk',fgmk)
    cv.imshow('frame',frame)

    if cv.waitKey(150) & 0xff ==27:
        break
cv.destroyAllWindows()