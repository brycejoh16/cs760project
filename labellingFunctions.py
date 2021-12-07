# --- Library Imports ----------------------------------------------------------
import numpy as np
from skimage.morphology import reconstruction
from scipy.signal import find_peaks
from skimage import filters

# --- Noisy Labelling Functions for MNIST dataset ------------------------------
def pixelCount(x): # counts pixels > 0 in the image
    image = x.reshape(28,28)
    count = 0
    for i in image:
        for j in i:
            if j > 0:
                count += 1
    return count

def fill(image):
    seed = x.copy()
    seed[1:-1, 1:-1] = x.max()
    mask = x
    filledImage = reconstruction(seed, mask, method='erosion')
    return filledImage

def fillCount(x): # fills enclosed parts and counts pixels > 0
    image = x.reshape(28,28)
    filledImage = fill(image)
    count = 0
    for i in filledImage:
        for j in i:
          if j > 0:
            count += 1
    return count

def fillSum(x): # fills enclosed parts and sums grayscale image
    image = x.reshape(28, 28)
    filled = fill(image)
    return np.sum(filled)

# calculates euclidean distance of each image from the _0_ matrix
# this is the L2 norm
def euclideanDistance(x):
  zeroImage = np.ones_like(x)*-1
  return np.sum((zeroImage-x)**2)

def euclideanBinarize(x):
    # this is super wierd b/c like it'c coming from
    x = x.copy()
    zeroImage = x.copy() * 0
    return np.sum((zeroImage - x) ** 2)

# counts peaks in count of pixels > 0 in one direction of the image
def peakCount(x):
    counts = []
    for i in x:
        count = 0
        for j in i:
            if j > 0:
                count += 1
        counts.append(count)
    numPeaks = len(find_peaks(counts)[0])
    return numPeaks

# counts peaks in the horizontal direction of the image
def horizontalPeakCount(x):
    image = x.reshape(28,28)
    return peakCount(image)

# counts peaks in the vertical direction of the image
def verticalPeakCount(x):
    image = x.reshape(28,28).T
    return peakCount(image)

# returns ratio of peaks in horizontal direction to peaks in vertical direction
def ratioPeakCount(x):
    vpc = verticalPeakCount(x)
    if vpc == 0:
        return horizontalPeakCount(x)
    else:
        return horizontalPeakCount(x)/vpc

# does roberts edge detection and counts peaks in the vertical direction
def edgeDetectVertical(x):
    image = x.reshape(28, 28)
    edgeImage = filters.roberts(image)
    return verticalPeakCount(edgeImage)

# does roberts edge detection and counts peaks in the horizontal direction
def edgeDetectHorizontal(x):
    image = x.reshape(28, 28)
    edgeImage = filters.roberts(image)
    return horizontalPeakCount(edgeImage)

# edge detection and returns ratio of peaks in horizontal and vertical direction
def edgeDetectRatio(x):
    image = x.reshape(28, 28)
    edgeImage = filters.roberts(image)
    return ratioPeakCount(edgeImage)
