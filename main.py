import cv2
from Utils import makeRandomPicture


def main():
    image = cv2.imread('Images/road.jpg')
    print('image shape={}, size={}'.format(image.shape, image.size))
    image[:, :, 1] = 0
    cv2.imshow('Image without green channel', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    makeRandomPicture()


if __name__ == "__main__":
    print('Hello, Open Source Computer Vision(OpenCV)')
    main()
