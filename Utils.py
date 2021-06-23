import cv2
import numpy
import os
from scipy import ndimage, interpolate


def makeRandomPicture():
    randomByte = bytearray(os.urandom(120000))
    flatNumpyArray = numpy.array(randomByte)

    grayImage = flatNumpyArray.reshape(300, 400)
    cv2.imwrite('Images/RandomGray.png', grayImage)

    bgrImage = flatNumpyArray.reshape(100, 400, 3)
    cv2.imwrite('Images/RandomBGR.png', bgrImage)


def passFilter(imageFile):
    kernel_3x3 = numpy.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
    kernel_5x5 = numpy.array([[-1, -1, -1, -1, -1],
                              [-1, 1, 2, 1, -1],
                              [-1, 2, 4, 2, -1],
                              [-1, 1, 2, 1, -1],
                              [-1, -1, -1, -1, -1]])
    img = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)

    k3 = ndimage.convolve(img, kernel_3x3)
    k5 = ndimage.convolve(img, kernel_5x5)

    blurred = cv2.GaussianBlur(img, (11, 11), 0)  # 高斯模糊
    g_hpf = img - blurred

    cv2.imshow('3x3', k3)
    cv2.imshow('5x5', k5)
    cv2.imshow('g_hpf', g_hpf)
    cv2.waitKey()
    cv2.destroyAllWindows()


def cannyPicture(imageFile):
    im = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('Images/canny.jpg', cv2.Canny(im, 200, 300))
    cv2.imshow('Canny', cv2.imread('Images/canny.jpg'))
    cv2.waitKey()
    cv2.destroyAllWindows()


def imageContours():
    img = numpy.zeros((200, 200), dtype=numpy.uint8)
    img[50:150, 50:150] = 255

    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contours", color)
    cv2.waitKey()
    cv2.destroyAllWindows()


def hammerContours():
    img = cv2.pyrDown(cv2.imread('Images/hammer.jpg', cv2.IMREAD_UNCHANGED))

    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        print('contour = {}'.format(c))
        # find bounding box coordinates
        x, y, w, h = cv2.boundingRect(c)
        print(x, y, w, h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # find minimum area
        rect = cv2.minAreaRect(c)
        # calculate coordinate of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = numpy.int0(box)
        # draw contour
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

        # calculate center and radius of minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # cast to integers
        center = (int(x), int(y))
        radius = int(radius)
        # draw the circle
        img = cv2.circle(img, center, radius, (0, 255, 0), 2)

    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    cv2.imshow('contours', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def Approx():
    img = cv2.pyrDown(cv2.imread("Images/hammer.jpg", cv2.IMREAD_UNCHANGED))

    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    black = cv2.cvtColor(numpy.zeros((img.shape[1], img.shape[0]), dtype=numpy.uint8), cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 计算近似多边形框
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 获取处理过的轮廓信息
        hull = cv2.convexHull(cnt)
        cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(black, [approx], -1, (255, 255, 0), 2)
        cv2.drawContours(black, [hull], -1, (0, 0, 255), 2)

    cv2.imshow("hull", black)
    cv2.waitKey()
    cv2.destroyAllWindows()


def lineDetect():
    img = cv2.imread('Images/tyre.bmp')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    minLineLength = 10
    maxLineGap = 3
    lines = cv2.HoughLinesP(edges, 1, numpy.pi / 180, 100, minLineLength, maxLineGap)
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
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120, param1=150, param2=30, minRadius=90, maxRadius=180)
    circles = numpy.uint16(numpy.around(circles))

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imwrite('Images/planets_circles.jpg', planets)
    cv2.imshow('HoughCircles', planets)
    cv2.waitKey()
    cv2.destroyAllWindows()




