import cv2
import numpy
import time


class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self.__capture = capture
        self.__channel = 0
        self.__enteredFrame = False
        self.__frame = None
        self.__imageFilename = None
        self.__videoFilename = None
        self.__videoEncoding = None
        self.__videoWriter = None

        self.__startTime = None
        self.__framesElapsed = int(0)
        self.__fpsEstimated = None

    @property
    def channel(self):
        return self.__channel

    @channel.setter
    def channel(self, value):
        if value != self.__channel:
            self.__channel = value
            self.__frame = None

    @property
    def frame(self):
        if self.__enteredFrame and self.__frame is None:
            _, self.__frame = self.__capture.retrieve()
        return self.__frame

    @property
    def isWritingImage(self):
        return self.__imageFilename is not None

    @property
    def isWritingVideo(self):
        return self.__videoFilename is not None

    def enterFrame(self):
        """ Capture the next frame, if any """

        # But first, check that any previous frame was exited.
        assert not self.__enteredFrame, 'previous enterFrame() had no matching exitFrame()'

        if self.__capture is not None:
            self.__enteredFrame = self.__capture.grab()

    def exitFrame(self):
        """ Draw to the window. Write to files. Release the frame. """

        # Check whether any grabbed frame is retrievable
        # The getter may retrieve and cache the frame
        if self.frame is None:
            self.__enteredFrame = False
            return

        # Update the FPS estimate and related variables
        if self.__framesElapsed == 0:
            self.__startTime = time.time()
        else:
            timeElapsed = time.time() - self.__startTime
            self.__fpsEstimated = self.__framesElapsed / timeElapsed
        self.__framesElapsed += 1

        # Draw to the window, if any
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = numpy.fliplr(self.__frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self.__frame)

        # Write to the image file, if any
        if self.isWritingImage:
            cv2.imwrite(self.__imageFilename, self.__frame)
            self.__imageFilename = None

        # Write to the video file, if any
        self._writeVideoFrame()

        # Release the frame
        self.__frame = None
        self.__enteredFrame = False

    def writeImage(self, filename):
        """ Write the next exited frame to an image file """

        self.__imageFilename = filename

    def startWritingVideo(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        """ Start writing exited frame to a video file """

        self.__videoFilename = filename
        self.__videoEncoding = encoding

    def stopWritingVideo(self):
        """ Stop writing exited frame to a video file """
        self.__videoFilename = None
        self.__videoEncoding = None
        self.__videoWriter = None

    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return

        if self.__videoWriter is None:
            fps = self.__capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self.__framesElapsed < 20:
                    # Wait until more frames elapse so that estimate is more stable
                    return
                else:
                    fps = self.__fpsEstimated
            size = (int(self.__capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.__capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.__videoWriter = cv2.VideoWriter(self.__videoFilename, self.__videoEncoding, fps, size)

        self.__videoWriter.write(self.__frame)


class WindowManager(object):
    def __init__(self, windowName, keypressCallback=None, mouseCallback=None):
        self.keypressCallback = keypressCallback
        self.mouseCallback = mouseCallback

        self.__windowName = windowName
        self.__isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self.__isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self.__windowName)
        cv2.setMouseCallback(self.__windowName, self.mouseCallback)
        self.__isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self.__windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self.__windowName)
        self.__isWindowCreated = False

    def processEvent(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            # Discard any non-ASCII info encoded by GTK
            keycode &= 0xFF
            self.keypressCallback(keycode)
