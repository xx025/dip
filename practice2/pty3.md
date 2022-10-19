# Practice 2:

## 实践内容：

### 1. 灰度图像的直方图均衡化处理；

**实现代码：**

```python

from copy import deepcopy
import cv2
import matplotlib
from matplotlib import pyplot as plt
from numpy import histogram
matplotlib.use('Qt5Agg')

class ImgHiramEqa:
    def __init__(self, img):
        self.img = img
        self.fx = self.crete_fx()
    def crete_fx(self):
        points = self.img.ravel()
        ll = histogram(points, bins=[i for i in range(257)])[0]
        l2 = [None for _ in range(len(ll))]
        for i1, j1 in enumerate(ll):
            l2[i1] = sum(ll[:i1 + 1]) / len(points)
        fx = {}
        # 建立映射关系
        for index, j in enumerate(l2):
            fx[str(index)] = int(j * 255 + 0.5)
        return fx

    def cort_img(self):
        img2 = deepcopy(self.img)
        # 进行像素映射处理
        for key, val in self.fx.items():
            img2[img == int(key)] = val
        return img2
```

**测试样例**

```python

path = "../practice2/leuvenA_s.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img2 = ImgHiramEqa(img=img).cort_img()
plt.hist(img.ravel(), 256)
plt.hist(img2.ravel(), 256)
cv2.imshow('img', img)
cv2.imshow('img2', img2)
plt.show()
```

**测试结果**

| 原                                                           | 处理后的图                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="imgs/leuvenA_s.jpg" alt="leuvenA_s" style="zoom: 50%;" /> | <img src="imgs/image-20221018174659125.png" alt="image-20221018174659125" style="zoom:50%;" /> |



直方图

> 蓝色为原始直方图，橘色为处理后的图像的直方图

<img src="imgs/Figure_1.png" alt="Figure_1" style="zoom: 67%;" />



### 2. 图像空域低通滤波；

### 3. 空域低通滤波消除噪声；

### 4. 空域高通滤波；

### 5. 图像的锐化处理；

**实现代码**

```python
import cv2
import numpy


class Laplace:
    m1 = [[0, -1, 0],
          [-1, 4, -1],
          [0, -1, 0]]
    m2 = [[-1, -1, -1],
          [-1, 8, -1],
          [-1, -1, -1]]


class ImgSharpen:

    def __init__(self, img: numpy.ndarray, mould: list):
        self.img = img
        self.mould = numpy.array(mould)
        self.Ishape = self.img.shape
        self.Mshape = self.mould.shape
        self.img_outline = self.filter()
        self.fil_tran_img = self.translation()
    def filter(self):
        """
        进行滤波操作
        :return:
        """
        _img_ = numpy.zeros(self.Ishape, dtype='uint8')
        for i in range(self.Ishape[0] - self.Mshape[0]):
            for j in range(self.Ishape[1] - self.Mshape[1]):
                m1 = self.img[i:i + self.Mshape[0], j: j + self.Mshape[1]]
                val = numpy.sum(m1 * self.mould)
                _img_[i + 1][j + 1] = 0 if val < 0 else val
        return _img_
    def translation(self):
        """
        进行灰度平移
        :return:
        """
        t_img = self.img + self.img_outline
        return numpy.array(t_img, dtype='uint8')
```

**测试样例**

```python
path = "../opencv-samples-data/lena.jpg"
img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img2 = ImgSharpen(img=img1, mould=Laplace.m2)
cv2.imshow('img', img1)
cv2.imshow('img2', img2.img_u)
cv2.imshow('img3', img2.img_sharpen)
cv2.waitKey()
```

**测试结果**

| 原图                                                         | 轮廓                                                         | 锐化后的图                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="imgs/image-20221019141709464.png" alt="image-20221019141709464" style="zoom:33%;" /> | <img src="imgs/image-20221019141731464.png" alt="image-20221019141731464" style="zoom:33%;" /> | <img src="imgs/image-20221019141850111.png" alt="image-20221019141850111" style="zoom:33%;" /> |

### 6. 椒盐噪声消除处理。

**实现代码**

```python
import cv2
import numpy
from numpy import median


class FilterMould:
    m1 = [[1, 1, 0, 1, 1],
          [1, 1, 0, 1, 1],
          [0, 0, 0, 0, 0],
          [1, 1, 0, 1, 1],
          [1, 1, 0, 1, 1]]

    m2 = [[0, 1, 1, 1, 0],
          [1, 0, 1, 0, 1],
          [1, 1, 0, 1, 1],
          [1, 0, 1, 0, 1],
          [0, 1, 1, 1, 0]]

    m3 = [[1, 0, 1, 0, 1],
          [0, 1, 1, 1, 0],
          [1, 1, 0, 1, 1],
          [0, 1, 1, 1, 0],
          [1, 1, 1, 1, 1]]


class FilterMethod:
    max_filter = 'max'
    min_filter = 'min'
    center_filter = 'center'
    medium_filter = 'medium'


class ImgFilter:

    def __init__(self, img: numpy.ndarray, mould, method: str):
        """
        
        :param img: 原始图像 
        :param mould: 滤波模板
        :param method: 滤波方式
        """

        self.img = img
        self.method = method
        self.mould = numpy.array(mould)
        self.shape_img = self.img.shape
        self.Mshape = self.mould.shape
        self.img_u = self.filter()

    def filter(self):
        """
        进行滤波操作
        :return:
        """
        _img_ = numpy.copy(self.img)
        r = (self.Mshape[0]) // 2 + 1
        m = self.Mshape[0]

        def medium_filter():
            for i in range(self.shape_img[0] - m):
                for j in range(self.shape_img[1] - m):
                    m1 = self.img[i:i + m, j: j + m]
                    _img_[i + r][j + r] = median(m1 * self.mould)

        def max_filter():
            for i in range(self.shape_img[0] - m):
                for j in range(self.shape_img[1] - m):
                    m1 = self.img[i:i + m, j: j + m]
                    _img_[i + r][j + r] = numpy.max(m1 * self.mould)

        def min_filter():
            for i in range(self.shape_img[0] - m):
                for j in range(self.shape_img[1] - m):
                    m1 = self.img[i:i + m, j: j + m]
                    _img_[i + r][j + r] = numpy.min(m1 * self.mould)

        def center_filter():
            for i in range(self.shape_img[0] - m):
                for j in range(self.shape_img[1] - m):
                    m1 = self.img[i:i + m, j: j + m]
                    val = numpy.min(m1 * self.mould) + numpy.min(m1 * self.mould)
                    _img_[i + r][j + r] = val // 2

        if self.method == 'max':
            max_filter()
        elif self.method == 'min':
            min_filter()
        elif self.method == 'center':
            center_filter()
        elif self.method == 'medium':
            medium_filter()
        else:
            medium_filter()
        return _img_
```

**测试样例**

```python
path = "../practice2/imgelpa.png"
img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img2 = ImgFilter(img=img1,
                 mould=FilterMould.m1,
                 method=FilterMethod.medium_filter)
cv2.imshow('img', img1)
cv2.imshow('img2', img2.img_u)

cv2.waitKey()


```

**测试结果**

| 原图                                                         | 中值滤波后的图                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="imgs/image-20221019162504533.png" alt="image-20221019162504533"  /> | ![image-20221019164837182](imgs/image-20221019164837182.png) |





**汇报人：** 

