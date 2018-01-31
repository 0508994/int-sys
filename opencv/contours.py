import numpy as np
import cv2
import argparse


def detect_contours(path, min_t=1, max_t=255):
    im = cv2.imread(path)
    blurred = cv2.GaussianBlur(im, (3, 3), 0)
    imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    


#   Binary Threshold
#   https://www.learnopencv.com/opencv-threshold-python-cpp/
#   if src(x,y) > thresh # - prvi argument funkcije posle slike ->
#       dst(x,y) = maxValue # - drugi argument funkcije posle slike
#   else
#       dst(x,y) = 0 

    # 1 - crna pozadina !
    thresh = cv2.threshold(imgray, min_t, max_t, cv2.THRESH_BINARY)[1]


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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('min')
    parser.add_argument('max')
    args = parser.parse_args()

    detect_contours(args.path, int(args.min), int(args.max))


if __name__ == '__main__':
    main()
