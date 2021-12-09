# --- Imports ------------------------------------------------------------------
import numpy as np
from skimage.morphology import reconstruction
from scipy.signal import find_peaks
from scipy.fft import fft,fftn
from skimage import filters
from snorkel.labeling import labeling_function

# --- Noisy labeling Functions for MNIST dataset -------------------------------

# Continuous variable functions and their helper functions return continuous
# variables for classifying MNIST data, Snorkel labeling functions return class
# labels based on continuous variables and are prefixed classify.

# counts nonempty (greater than lowestValue) pixels in the image array x
def pixelCount(x, lowestValue=-1, axis=None):
    return np.count_nonzero(x > lowestValue, axis=axis)

@labeling_function()
def classifyPixelCount(x):
    # Return a label of 0 if pixelCount() < threshold, otherwise 1
    threshold = 130 # threshold between 0 and 1
    var = pixelCount(x)

    if var < threshold:
        return 1
    else:
        return 0

def fill(image): # fills enclosed areas in image and returns filled image
    seed = image.copy()
    seed[1:-1, 1:-1] = image.max()
    mask = image
    filledImage = reconstruction(seed, mask, method='erosion')
    return filledImage

def fillCount(x,lowestValue=-1): # fills enclosed parts and returns count of nonempty pixels
    image = x.reshape(28,28)
    filledImage = fill(image)
    count = pixelCount(filledImage,lowestValue=lowestValue)
    return count

@labeling_function()
def classifyFillCount(x):
    # Return a label of 0 if fillCount() < threshold, otherwise 1
    threshold = 150 # threshold between 0 and 1
    var = fillCount(x)

    if var < threshold:
        return 1
    else:
        return 0

def fillSum(x): # fills enclosed parts and sums grayscale image
    image = x.reshape(28, 28)
    filled = fill(image)
    return np.sum(filled)

@labeling_function()
def classifyFillSum(x):
    # Return a label of 0 if fillSum() < threshold, otherwise 1
    threshold = -550 # threshold between 0 and 1
    var = fillSum(x)

    if var < threshold:
        return 1
    else:
        return 0

def l2Norm(x): # returns the L2 norm of the image
    image = x.reshape(28, 28)
    return np.linalg.norm(image)

@labeling_function()
def classifyL2Norm(x):
    # Return a label of 0 if l2Norm() < threshold, otherwise 1
    threshold = 27.1 # threshold between 0 and 1
    var = l2Norm(x)

    if var < threshold:
        return 0
    else:
        return 1

# returns peak count in nonempty pixel count in one direction of the image
def peakCount(x, lowestValue, height, prominence, axis):
    counts = pixelCount(x, lowestValue, axis)
    numPeaks = len(find_peaks(counts, height=height, prominence=prominence)[0])
    return numPeaks

# returns peak count in the horizontal direction
def horizontalPeakCount(x, lowestValue=-1):
    height = 7
    prominence = None
    image = x.reshape(28,28)
    return peakCount(image, lowestValue, height, prominence, axis=0)

@labeling_function()
def classifyHorizontalPeakCount(x):
    # Return a label of 0 if horizontalPeakCount() < threshold, otherwise 1
    threshold = 1.5 # threshold between 0 and 1
    var = horizontalPeakCount(x)

    if var < threshold:
        return 1
    else:
        return 0

# returns peak count in the vertical direction
def verticalPeakCount(x, lowestValue=-1):
    height = None
    prominence = 8
    image = x.reshape(28,28)
    return peakCount(image, lowestValue, height, prominence, axis=1)

@labeling_function()
def classifyVerticalPeakCount(x):
    # Return a label of 0 if verticalPeakCount() < threshold, otherwise 1
    threshold = 0.5 # threshold between 0 and 1
    var = verticalPeakCount(x)

    if var < threshold:
        return 1
    else:
        return 0

# returns ratio of peaks in horizontal direction to peaks in vertical direction
def ratioPeakCount(x, lowestValue=-1):
    vpc = verticalPeakCount(x, lowestValue)
    if vpc == 0:
        return horizontalPeakCount(x, lowestValue)
    else:
        return horizontalPeakCount(x, lowestValue)/vpc

@labeling_function()
def classifyRatioPeakCount(x):
    # Return a label of 0 if ratioPeakCount() < threshold, otherwise 1
    threshold = 1.5 # threshold between 0 and 1
    var = ratioPeakCount(x)

    if var < threshold:
        return 1
    else:
        return 0

# does roberts edge detection and counts peaks in the vertical direction
def edgeDetectVertical(x):
    image = x.reshape(28, 28)
    edgeImage = filters.roberts(image)
    return verticalPeakCount(edgeImage, lowestValue=0)

@labeling_function()
def classifyEdgeDetectVertical(x):
    # Return a label of 0 if edgeDetectVertical() < threshold, otherwise 1
    threshold = 0.5 # threshold between 0 and 1
    var = edgeDetectVertical(x)

    if var < threshold:
        return 1
    else:
        return 0

# does roberts edge detection and counts peaks in the horizontal direction
def edgeDetectHorizontal(x):
    image = x.reshape(28, 28)
    edgeImage = filters.roberts(image)
    return horizontalPeakCount(edgeImage, lowestValue=0)

@labeling_function()
def classifyEdgeDetectHorizontal(x):
    # Return a label of 0 if edgeDetectHorizontal() < threshold, otherwise 1
    threshold = 2.5 # threshold between 0 and 1
    var = edgeDetectHorizontal(x)

    if var < threshold:
        return 1
    else:
        return 0

# edge detection and returns ratio of peaks in horizontal and vertical direction
def edgeDetectRatio(x):
    image = x.reshape(28, 28)
    edgeImage = filters.roberts(image)
    return ratioPeakCount(edgeImage, lowestValue=0)

@labeling_function()
def classifyEdgeDetectRatio(x):
    # Return a label of 0 if edgeDetectRatio() < threshold, otherwise 1
    threshold = 1.5 # threshold between 0 and 1
    var = edgeDetectRatio(x)

    if var < threshold:
        return 1
    else:
        return 0


def fourier_Transorm(x):
    yf=fft(x)
    c=20
    #sum of the first 20 fourier coeficients
    return np.sum(2.0 /(28*28) * np.abs(yf[0:c]))


def fft_ndimensions(x):
    x=x.reshape(28,28)
    yf=fftn(x)
    # take first 10 coeficients again.
    return np.sum(np.abs(yf)[:10,:10])

def log_regression(x):
    # okay still need to do the noisy logistic regression.
    pass


if __name__ == '__main__':
    fft_ndimensions(np.random.random((28*28)))