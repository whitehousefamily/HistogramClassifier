import numpy
import cv2
import scipy.io
import scipy.sparse

class HistrogramClassifier(object):

    def __init__(self):
        self.verbose = False
        self.minimumSimilarityForPositiveLabel = 0.075

        self._channels = range(3)
        self._histSize = [256] * 3
        self._ranges = [0, 255] * 3
        self._references = {}

    def _createNormalizedHist(self, image, sparse):
        # Create the histogram
        hist = cv2.calcHist([image], self._channels, None,
                            self._histSize, self._ranges)
        # Normalise the histogram
        hist[:] = hist * (1.0 / numpy.sum(hist))
        # Convert the histogram to one column for efficient storage
        hist = hist.reshape(16777216, 1)
        if sparse:
            # convert the histogram to a sparse matrix
            hist = scipy.sparse.csc_matrix(hist)
        return hist

    
