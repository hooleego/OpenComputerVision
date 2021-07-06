import cv2
import numpy
from scipy import ndimage
from Utils import hammerContours, approx, lineDetect, circleDetect


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

    # 高通滤波器（HPF）：检测图像的某个区域，然后根据像素与周围像素的亮度差值来提神该像素的亮度
    k3 = ndimage.convolve(img, kernel_3x3)
    k5 = ndimage.convolve(img, kernel_5x5)
    # 低通滤波器（LPF）：在像素与周围像素的亮度差值小于一个特定值时，平滑该像素的亮度，主要用于去噪和模糊化
    blurred = cv2.GaussianBlur(img, (7, 7), 0)  # 高斯模糊
    hpf = img - blurred

    cv2.imshow('3x3', k3)
    cv2.imshow('5x5', k5)
    cv2.imshow('High Pass Filter', hpf)
    cv2.waitKey()
    cv2.destroyAllWindows()


def cannyEdge(imageFile):
    """ 使用OpenCV的Canny进行图像的边缘检测 """

    img = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    canny_img = cv2.Canny(img, 200, 300)
    cv2.imshow('Canny', canny_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def imageContours(imageFile):
    img = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (13, 13), 0)
    ret, thresh = cv2.threshold(blurred, 154, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold', thresh)
    cv2.waitKey()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color, contours, -1, (255, 0, 0), 1)
    cv2.imshow("Contours", color)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    image = cv2.imread('Images/river.jpg')
    print('image shape={}, size={}'.format(image.shape, image.size))
    image[:, :, 1] = 0
    cv2.imshow('Image without green channel', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('Hello, Open Source Computer Vision(OpenCV)')
    # main()
    # passFilter('Images/river.jpg')
    # cannyEdge('Images/river.jpg')
    # imageContours('Images/river.jpg')
    # hammerContours()
    # approx()
    # lineDetect()
    circleDetect()
