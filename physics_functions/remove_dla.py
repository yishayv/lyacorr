import numpy as np
from scipy import signal


class RemoveDlaSimple(object):
    def __init__(self):
        self.detect_box_size = 30
        self.detect_boxcar = signal.boxcar(self.detect_box_size)
        self.mask_box_size = 60
        self.mask_boxcar = signal.boxcar(self.mask_box_size)

    def get_mask(self, ar_flux):
        # detect low flux regions using threshold
        # convolve and divide by box_size to keep the same scale
        spec1_smooth = signal.convolve(ar_flux > -0.8, self.detect_boxcar, mode='same') / self.detect_box_size

        # expand the mask to nearby pixels by smoothing
        mask_thresh = np.array(spec1_smooth < 0.2)
        mask_smooth = signal.convolve(mask_thresh, self.mask_boxcar, mode='same') > 1

        return mask_smooth
