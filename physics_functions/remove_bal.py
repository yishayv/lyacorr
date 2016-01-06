import numpy as np
from scipy import signal


class RemoveBalSimple(object):
    def __init__(self):
        self.box_size = 75
        self.boxcar = signal.boxcar(self.box_size)

    def get_mask(self, ar_flux):
        # convolve and divide by box_size to keep the same scale
        spec1_smooth = signal.convolve(ar_flux, self.boxcar, mode='same') / self.box_size

        # detect low flux regions using threshold
        mask_thresh = np.array(spec1_smooth < -0.5)
        # expand the mask to nearby pixels by smoothing
        mask_smooth = signal.convolve(mask_thresh, self.boxcar, mode='same') > 10

        return mask_smooth
