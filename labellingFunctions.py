# --- Library Imports ----------------------------------------------------------
import numpy as np
from skimage.morphology import reconstruction
from scipy.signal import find_peaks
from skimage import filters

# --- Noisy Labelling Functions for MNIST dataset ------------------------------
def fillCount(x): # fills enclosed parts and counts pixels > 0
    x=x.reshape(28,28)
    seed =x.copy()
    seed[1:-1, 1:-1] = x.max()
    mask = x

    filled = reconstruction(seed, mask, method='erosion')
    count = 0
    for i in filled:
        for j in i:
          if j > 0:
            count += 1
    return count

def fillSum(x): # fills enclosed parts and sums grayscale image
    x = x.reshape(28, 28)
    seed =x.copy()
    seed[1:-1, 1:-1] = x.max()
    mask = x
    filled = reconstruction(seed, mask, method='erosion')
    return np.sum(filled)

def pixelCount(x): # counts pixels > 0 in the image
    image=x.reshape(28,28)
    count = 0
    for i in image:
        for j in i:
            if j > 0:
                count += 1
    return count

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



# counts peaks in the horizontal direction of the image
def horizontalPeakCount(x):
    x=x.reshape(28,28)
    counts = []
    for i in x:
        count = 0
        for j in i:
            if j > 0:
                count += 1
        counts.append(count)
    return len(find_peaks(counts)[0])

# counts peaks in the vertical direction of the image
def verticalPeakCount(x):
    image=x.reshape(28,28).T
    counts = []
    # image = np.copy(image).transpose()
    for i in image:
        count = 0
        for j in i:
            if j > 0:
                count += 1
        counts.append(count)
    return len(find_peaks(counts)[0])

# returns ratio of peaks in horizontal direction to peaks in vertical direction
def ratioPeakCount(image):
  vpc = verticalPeakCount(image)
  if vpc == 0:
    return horizontalPeakCount(image)
  else:
    return horizontalPeakCount(image)/vpc

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
