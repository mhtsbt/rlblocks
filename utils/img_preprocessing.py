import cv2


class Utils:

    @staticmethod
    def greyscale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def resize(img, height, width):
        return cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)
