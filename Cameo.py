import cv2
from Managers import CaptureManager, WindowManager
import Filters


class Cameo(object):
    def __init__(self, videoSource):
        self.__windowManager = WindowManager('Cameo', self.onKeypress, self.onMouse)
        self.__capture = cv2.VideoCapture(videoSource)
        self.__captureManager = CaptureManager(self.__capture, self.__windowManager, True)
        self.__leftButtonClicked = False
        self.__curveFilter = Filters.BlurFilter()

    def run(self):
        """ Run the main loop """

        self.__windowManager.createWindow()
        while self.__windowManager.isWindowCreated:
            if self.__leftButtonClicked:
                self.__windowManager.processEvent()
            else:
                self.__captureManager.enterFrame()
                frame = self.__captureManager.frame
                Filters.strokeEdges(frame, frame)
                self.__curveFilter.apply(frame, frame)
                self.__windowManager.show(frame)
                self.__captureManager.exitFrame()
                self.__windowManager.processEvent()
        self.__capture.release()

    def onKeypress(self, keycode):
        """ Handle a keypress

        space  -> Take a screenshot
        tab    -> Start/stop recording a screencast
        escape -> Quit
        """
        if keycode == 32:  # space
            self.__captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # tab
            if not self.__captureManager.isWritingVideo:
                self.__captureManager.startWritingVideo('screencast.avi')
            else:
                self.__captureManager.stopWritingVideo()
        elif keycode == 27:  # escape
            self.__windowManager.destroyWindow()

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.__leftButtonClicked = not self.__leftButtonClicked


if __name__ == "__main__":
    Cameo(0).run()

