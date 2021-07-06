import cv2
import numpy
import os


def makeRandomPicture():
    randomByte = bytearray(os.urandom(120000))
    flatNumpyArray = numpy.array(randomByte)

    grayImage = flatNumpyArray.reshape(300, 400)
    cv2.imwrite('Images/RandomGray.png', grayImage)

    bgrImage = flatNumpyArray.reshape(100, 400, 3)
    cv2.imwrite('Images/RandomBGR.png', bgrImage)


def hammerContours():
    img = cv2.pyrDown(cv2.imread('Images/hammer.jpg', cv2.IMREAD_UNCHANGED))
    # 模糊化处理，防止将噪声识别为轮廓
    blurred = cv2.GaussianBlur(img, (23, 23), 0)
    # 二值化操作
    ret, thresh = cv2.threshold(cv2.cvtColor(blurred.copy(), cv2.COLOR_BGR2GRAY), 161, 255, cv2.THRESH_BINARY)
    cv2.imshow('Hammer Threshold', thresh)
    # 轮廓检测
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours), contours)
    for c in contours:
        print('contour = {}'.format(c))
        # find bounding box coordinates
        x, y, w, h = cv2.boundingRect(c)
        print('Bounding rect = {}'.format((x, y, w, h)))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 计算最小矩形
        rect = cv2.minAreaRect(c)
        print('Minimum rect = {}'.format(rect))
        # 计算最小矩形的顶点坐标（结果为浮点数）
        box = cv2.boxPoints(rect)
        print('Box point = {}'.format(box))
        # 将浮点坐标转为整形坐标
        box = numpy.int0(box)
        # draw contour
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        # calculate center and radius of minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # cast to integers
        center = (int(x), int(y))
        radius = int(radius)
        # draw the circle
        img = cv2.circle(img, center, radius, (255, 0, 0), 2)

    cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
    cv2.imshow('Hammer Contours', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def approx():
    """ 检测和绘制凸轮廓 """

    img = cv2.pyrDown(cv2.imread("Images/hammer.jpg", cv2.IMREAD_UNCHANGED))
    # 模糊化处理，防止将噪声识别为轮廓
    blurred = cv2.GaussianBlur(img, (23, 23), 0)
    # 二值化操作
    ret, thresh = cv2.threshold(cv2.cvtColor(blurred.copy(), cv2.COLOR_BGR2GRAY), 161, 255, cv2.THRESH_BINARY)
    cv2.imshow('Hammer Threshold', thresh)
    print(img.shape[1], img.shape[0])
    black = cv2.cvtColor(numpy.zeros((img.shape[1], img.shape[0]), dtype=numpy.uint8), cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours), contours)

    for cnt in contours:
        # 计算近似多边形框
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        rox = cv2.approxPolyDP(cnt, epsilon, True)
        # 获取处理过的轮廓信息
        hull = cv2.convexHull(cnt)
        cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(black, [rox], -1, (255, 255, 0), 2)
        cv2.drawContours(black, [hull], -1, (0, 0, 255), 2)

    cv2.imshow("Hull", black)
    cv2.waitKey()
    cv2.destroyAllWindows()


def lineDetect():
    img = cv2.imread('Images/hammer.jpg')
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 15)
    # 边缘检测
    edges = cv2.Canny(blurred, 200, 300)
    minLineLength = 1
    maxLineGap = 15
    lines = cv2.HoughLinesP(edges, 1, numpy.pi / 180, 10, minLineLength, maxLineGap)
    print(len(lines), lines)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Edges', edges)
    cv2.imshow('Lines', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def circleDetect():
    planets = cv2.imread('Images/planets.jpg')
    gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray_img, 5)
    cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120, param1=150, param2=30, minRadius=90, maxRadius=180)
    circles = numpy.uint16(numpy.around(circles))
    print(len(circles), circles)
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 2)

    cv2.imshow('HoughCircles', planets)
    cv2.waitKey()
    cv2.destroyAllWindows()




