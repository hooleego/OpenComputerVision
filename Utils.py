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


