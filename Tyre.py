import numpy as np
import cv2 as cv
import os


def getTopPoint(picturePath, x, y, height, width):
    """
    从图片中找出轮眉最高点位置的坐标

    :param picturePath: 图片文件路径
    :param x: ROI原点位于图片上的X坐标
    :param y: ROI原点位于图片上的Y坐标
    :param height: ROI区域高度
    :param width: ROI区域宽度
    :return: 返回（success, [x, y]）元组，其中success表示成功或失败，[x, y]表示轮眉最高点的x, y坐标
    """
    try:
        img = cv.imread(picturePath)
        """
        转灰度图
        """
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        save_path = r'{0[0]}\gray-{0[1]}'.format(os.path.split(picturePath))
        cv.imwrite(save_path, gray)

        """
        截取ROI区域
        """
        roi = gray[y:y + height, x:x + width]
        save_path = r'{0[0]}\roi-{0[1]}'.format(os.path.split(picturePath))
        cv.imwrite(save_path, roi)

        """
        双边滤波
        双边滤波是一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折衷处理，同时考虑空间与信息和灰度相似性，达到保边去噪的目的，
        具有简单、非迭代、局部处理的特点。
        src: 输入图像
        d: 过滤时周围每个像素领域的直径
        sigmaColor: 在color space中过滤sigma。参数越大，邻近像素将会在越远地方mix
        sigmaSpace: 在coordinate space中过滤sigma。参数越大，那些颜色足够相近的颜色的影响越大
        """
        bf_image = cv.bilateralFilter(roi, 5, 175, 175)
        save_path = r'{0[0]}\filter-{0[1]}'.format(os.path.split(picturePath))
        cv.imwrite(save_path, bf_image)

        """
        构造一个15*15，并且值全为1的二维列表
        """
        kernel = np.ones((15, 15), np.uint8)

        """
        opencv中的morphologyEx()函数时一种形态学变化函数。数学形态学可以理解为一种滤波行为，因此也称为形态学滤波。滤波中用到的滤波器(kernel)
        ，在形态学中称为结构元素。结构元素往往是由一个特殊的形态构成，如线条、矩形、圆等。
        src: 输入图像
        op: 运算符。cv.MORPH_OPEN表示开运算。开运算可以用来消除小黑点。
        kernel: 结构元素，这里我理解为是一个矩形
        """
        morph_open = cv.morphologyEx(bf_image, cv.MORPH_OPEN, kernel)
        save_path = r'{0[0]}\morph-{0[1]}'.format(os.path.split(picturePath))
        cv.imwrite(save_path, morph_open)

        """
        自适应阈值二值化
        src: 需要进行二值化的灰度图像
        maxValue: 满足条件的像素点需要设置的灰度值
        adaptiveMethod: 自适应阈值算法
        thresholdType: opencv提供的二值化方法
        blockSize: 要分成的区域大小，与形态学滤波的结构大小一致
        C: 常数，每个区域计算出的阈值的基础上再减去这个常数作为这个区域的最终阈值，可以为负数

        blockSize越大，参与计算阈值的区域也越大，细节轮廓就变得越少，整体轮廓越粗越明显
        当C越大，每个像素点的N*N邻域计算出的阈值就越小，中心点大于这个阈值的可能性就越大，设置成255的概率就越大，整体图像白色像素就越多
        """
        threshold = cv.adaptiveThreshold(morph_open, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 55, 5)
        save_path = r'{0[0]}\threshold-{0[1]}'.format(os.path.split(picturePath))
        cv.imwrite(save_path, threshold)

        """
        轮廓监测
        findContours函数接受的参数为二值图（即黑白的），所以读取的图像要先转成灰度的，再转成二值的
        image: 寻找轮廓的图像
        mode: 轮廓的检索模式，这里只监测外轮廓
        method: 轮廓的近似办法，这里压缩水平方向、垂直方向、对角线方向的元素，只保留改方向上的终点坐标
        """
        contours, hierarchy = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(roi, contours, -1, (255, 0, 255), 1)
        save_path = r'{0[0]}\contours-{0[1]}'.format(os.path.split(picturePath))
        cv.imwrite(save_path, roi)

        index = 0
        for contour in contours:
            """
            计算封闭轮廓的周长或曲线长度
            """
            val = cv.arcLength(contour, False)
            print('当前轮廓: {} -> {}'.format(contour, val))
            if val > width:
                break
            index = index + 1

        """
        每个轮廓的点是按从上到下排序的，因此第0个点就是最高点。这些点的是以(x,y)为坐标原点的坐标
        """
        max_x = contours[index][0][0]
        cv.line(img, (x, max_x[1] + y), (x + width, max_x[1] + y), (0, 0, 255))

        cv.rectangle(img, (x, y), (x + width, y + height), (255, 0, 255))
        save_path = r'{0[0]}\recognition-{0[1]}'.format(os.path.split(picturePath))
        cv.imwrite(save_path, img)
        return True, [max_x[0] + x, max_x[1] + y]
    except Exception as e:
        print(e)
        return False, [0, 0]


if __name__ == '__main__':
    # rl_path = 'Images/FR.bmp'
    # result, coordinate = getTopPoint(rl_path, 630, 190, 290, 850)

    # rl_path = 'Images/11.jpg'
    # result, coordinate = getTopPoint(rl_path, 1724, 677, 302, 1359)

    # rl_path = 'Images/12.jpg'
    # result, coordinate = getTopPoint(rl_path, 1597, 965, 329, 1212)

    # rl_path = 'Images/21.jpg'
    # result, coordinate = getTopPoint(rl_path, 1370, 902, 286, 1027)

    rl_path = 'Images/31.jpg'
    result, coordinate = getTopPoint(rl_path, 1737, 839, 253, 1188)

    print(result, coordinate)
