import numpy as np
import cv2


def detect_contours(path, min_t=1, max_t=255):
    im = cv2.imread(path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgray, (3, 3), 0)


#   Binary Threshold
#   https://www.learnopencv.com/opencv-threshold-python-cpp/
#   if src(x,y) > thresh # - prvi argument funkcije posle slike ->
#       dst(x,y) = maxValue # - drugi argument funkcije posle slike
#   else
#       dst(x,y) = 0 

    # 1 - crna pozadina !
    thresh = cv2.threshold(blurred, min_t, max_t, cv2.THRESH_BINARY)[1]


    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]


#   img = np.zeros(im.size).reshape(im.shape)


    # 1) image
    # 2) contours
    # 3) index of the contour
    # 4) color
    # 5) line-width
    cv2.drawContours(im, contours, -1, (0,255,0), 5)

    cv2.imshow('Detected contours', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    detect_contours('img/shapes1.png', 60, 255)
