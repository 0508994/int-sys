import sys
import numpy as np
import cv2

# https://stackoverflow.com/questions/23749968/why-datatype-has-to-be-uint8-in-opencv-python-wrapper 

def brightness(img, val, inplace=False):
    assert val > 0

#   uint8 overflow handling ...
#   e: 222 + 50 = 16
    limit = 255 - val
    img[img > limit] = 255
    img[img < limit] += val

    if not inplace:
        return img # ...

def contrast_slow(img, val, inplace=False):
    factor = (259 * (val + 255)) / (255 * (259 - val))

    h = img.shape[0]
    w = img.shape[1]
    for y in range(h):
        for x in range(w):
            blue = factor * (img.item(y, x, 2) - 128) + 128
            green = factor * (img.item(y, x, 1) - 128) + 128
            red = factor * (img.item(y, x, 0) - 128) + 128
            
            if blue > 255: blue = 255
            if blue < 0: blue = 0

            if green > 255: green = 255
            if green < 0: green = 0

            if red > 255: red = 255
            if red < 0: red = 0

            img.itemset((y, x, 2), blue)
            img.itemset((y, x, 1), green)
            img.itemset((y, x, 0), red)

    if not inplace:
        return img

def contrast(img, val):
    img = img.astype(np.int32)
    factor = (259 * (val + 255)) / (255 * (259 - val))

    img = factor * (img - 128) + 128
    img[img > 255] = 255
    img[img < 0] = 0

    return img.astype(np.uint8)

# cv2.Laplacian(src, ddepth, other_options...)
# where ddepth is the desired depth of the destination image

def edge(img, kernel='laplacian'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blured = cv2.GaussianBlur(gray, (3, 3), 0)
    #blured = gray
    if kernel == 'laplacian':
        return cv2.Laplacian(blured, cv2.CV_64F)
    elif kernel == 'sobelx':
        return cv2.Sobel(blured, cv2.CV_64F, 1, 0, ksize=5)
    else:
        return cv2.Sobel(blured, cv2.CV_64F, 0, 1, ksize=5)


def main():
#   filters.py <filter_name> <image_path> <arg>
#   e:python filters.py brightness img/bridge.jpg 129
    try:
        print(sys.argv)
        img = cv2.imread(sys.argv[2])
        if sys.argv[1] == 'brightness':
            img = brightness(img, int(sys.argv[3]))
        elif sys.argv[1] == 'contrast':
            img = contrast(img, int(sys.argv[3]))
        else:
            img = edge(img, kernel=sys.argv[3])
    except Exception as e:
        print(e)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()