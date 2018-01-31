# https://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv
import numpy as np
import cv2
import argparse
from scipy.spatial import distance as dist
from collections import OrderedDict
#import imutils



def detect_rects(path, min_t, max_t):
    im = cv2.imread(path)
    blurred = cv2.GaussianBlur(im, (3, 3), 0)
    imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    

    thresh = cv2.threshold(imgray, min_t, max_t, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    for cnt in contours:
#       In this, second argument is called epsilon, which is maximum distance from contour to approximated contour. It is an accuracy parameter.
#       A wise selection of epsilon is needed to get the correct output.
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            cv2.drawContours(im, [cnt], 0, (0, 0, 255), 4)


    cv2.imshow('Detected rectangles', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===============================================================================================================================================================
'''
    https://www.pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/

    So why are we using the L*a*b* color space rather than RGB or HSV?
    Well, in order to actually label and tag regions of an image as containing a certain color,
    we’ll be computing the Euclidean distance between our dataset of known colors (i.e., the lab  array)
    and the averages of a particular image region. The known color that minimizes the Euclidean distance will
    be chosen as the color identification. And unlike HSV and RGB color spaces,
    the Euclidean distance between L*a*b* colors has actual perceptual meaning.
'''
class ColorLabeler:
	def __init__(self):
		# initialize the colors dictionary, containing the color
		# name as the key and the RGB tuple as the value
		colors = OrderedDict({
			"red": (255, 0, 0),
			"green": (0, 255, 0),
			"blue": (0, 0, 255),
            "yellow": (234, 218, 71),
            "orange": (255, 160, 102),
            "green": (43, 185, 137),
            "blue2": (2, 132, 228),
            "purple": (153, 66, 219)})
 
		# allocate memory for the L*a*b* image, then initialize
		# the color names list
		self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
		self.colorNames = []
 
		# loop over the colors dictionary
		for (i, (name, rgb)) in enumerate(colors.items()):
			# update the L*a*b* array and the color names list
			self.lab[i] = rgb
			self.colorNames.append(name)
 
		# convert the L*a*b* array from the RGB color space
		# to L*a*b*
		self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

	def label(self, image, c):
		# construct a mask for the contour, then compute the
		# average L*a*b* value for the masked region
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		mask = cv2.erode(mask, None, iterations=2)
		mean = cv2.mean(image, mask=mask)[:3]
 
		# initialize the minimum distance found thus far
		minDist = (np.inf, None)
 
		# loop over the known L*a*b* color values
		for (i, row) in enumerate(self.lab):
			# compute the distance between the current L*a*b*
			# color value and the mean of the image
			d = dist.euclidean(row[0], mean)
 
			# if the distance is smaller than the current distance,
			# then update the bookkeeping variable
			if d < minDist[0]:
				minDist = (d, i)
 
		# return the name of the color with the smallest distance
		return self.colorNames[minDist[1]]

#=====================================================================================================================================================================

def detect_spec_rects(path, min_t, max_t, color_a, size):

    im = cv2.imread(path)
#   resized = imutils.resize(im, width=300)
    blurred = cv2.GaussianBlur(im, (3, 3), 0)
    imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    thresh = cv2.threshold(imgray, min_t, max_t, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    cl = ColorLabeler()

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        color = cl.label(lab, cnt)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and color == color_a and area >= size:
            print('Detected area size {}.'.format(area))
            cv2.drawContours(im, [cnt], 0, (255, 0, 255), 4)


    cv2.imshow('Detected rectangles', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def main():
#   python rectangles.py img/shapes.png 1 255 purple 0
#   python rectangles.py img/shapes1.png 60 255 green 0

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('min')
    parser.add_argument('max')
    parser.add_argument('color')
    parser.add_argument('size')
    args = parser.parse_args()

    detect_spec_rects(args.path, int(args.min), int(args.max), args.color, float(args.size))
    #detect_rects(args.path, int(args.min), int(args.max))


if __name__ == '__main__':
    main()