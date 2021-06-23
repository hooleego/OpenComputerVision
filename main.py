import cv2
from Utils import makeRandomPicture, passFilter, cannyPicture, imageContours, hammerContours, Approx, lineDetect, circleDetect
from Filters import strokeEdges


def main():
    image = cv2.imread('Images/river.jpg')
    print('image shape={}, size={}'.format(image.shape, image.size))
    image[:, :, 1] = 0
    cv2.imshow('Image without green channel', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # makeRandomPicture()

    # passFilter('Images/river.jpg')

    # cannyPicture('Images/tyre.bmp')

    # imageContours()

    # hammerContours()

    # Approx()

    # lineDetect()

    circleDetect()


if __name__ == "__main__":
    print('Hello, Open Source Computer Vision(OpenCV)')
    main()
