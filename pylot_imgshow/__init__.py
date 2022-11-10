from matplotlib import pyplot as plt


def img_show(*img, sp='0'):
    """
    :param img: 图片，（title,img）,...,() 顺序展示
    :param sp: 分割方式，默认全横排 或者 2#3 表示两行三列
    :return:
    """
    if sp == '0':
        sp = f"1#{len(img)}"
    sp = ''.join(sp.split("#"))

    for i in range(len(img)):
        plt.subplot(int(f"{sp}{i + 1}"))
        plt.imshow(img[i][1], cmap='gray'), plt.title(img[i][0])
        plt.xticks([]), plt.yticks([])
    plt.show()
