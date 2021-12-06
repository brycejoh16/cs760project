import numpy as np
from skimage.morphology import reconstruction
from scipy.signal import find_peaks
from skimage import filters
def fillCount(x):
    # fills enclosed parts and counts pixels > 0
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
def verticalPeakCount(image):
    counts = []
    image = np.copy(image).transpose()
    for i in image:
        count = 0
        for j in i:
            if j > 0:
                count += 1
        counts.append(count)
    return len(find_peaks(counts)[0])

# returns ratio of peaks in horizontal direction to peaks in vertical direction
def ratioPeakCount(image):
    ratio = horizontalPeakCount(image)/verticalPeakCount(image)
    return ratio

# does roberts edge detection and counts peaks in the vertical direction
def edgeDetectVertical(image):
    edgeImage = filters.roberts(image)
    return verticalPeakCount(edgeImage)

# does roberts edge detection and counts peaks in the horizontal direction
def edgeDetectHorizontal(image):
    edgeImage = filters.roberts(image)
    return horizontalPeakCount(edgeImage)

# edge detection and returns ratio of peaks in horizontal and vertical direction
def edgeDetectRatio(image):
    edgeImage = filters.roberts(image)
    return ratioPeakCount(edgeImage)
