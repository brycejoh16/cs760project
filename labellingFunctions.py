import numpy as np
from skimage.morphology import reconstruction
from scipy.signal import find_peaks
from skimage import filters

def fillCount(image): # fills enclosed parts and counts pixels > 0
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = np.copy(image)

    filled = reconstruction(seed, mask, method='erosion')
    count = 0
    for i in filled:
        for j in i:
          if j > 0:
            count += 1
    return count

 def fillSum(image): # fills enclosed parts and sums grayscale image
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = np.copy(image)
    filled = reconstruction(seed, mask, method='erosion')
    return np.sum(filled)

def pixelCount(image): # counts pixels > 0 in the image
    count = 0
    for i in np.copy(image):
        for j in i:
            if j > 0:
                count += 1
    return count

# calculates euclidean distance of each image from the _0_ matrix
def euclideanDistance(image):
  image = np.copy(image)
  zeroImage = image*0
  return np.sum((zeroImage-image)**2)

# counts peaks in the horizontal direction of the image
def horizontalPeakCount(image):
    counts = []
    for i in np.copy(image):
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
