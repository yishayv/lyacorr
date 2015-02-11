import numpy as np


class Bins2D:
    def __init__(self, x_count, y_count):
        self.ar_flux = np.zeros((x_count, y_count))
        self.ar_count = np.zeros((x_count, y_count))
        self.x_count = x_count
        self.y_count = y_count
        # pre-allocation:
        self.ar_x_int = np.zeros([5000, 5000], dtype='int64')
        self.ar_y_int = np.zeros([5000, 5000], dtype='int64')

    def add(self, flux, x, y):
        x_int = int(x)
        y_int = int(y)
        if x < 0 or y < 0 or x >= self.x_count or y >= self.y_count:
            return
        self.ar_flux[x, y] += flux
        self.ar_count[x, y] += 1

    def add_array(self, ar_flux, ar_x, ar_y):
        ar_x_int = ar_x.astype(int)
        ar_y_int = ar_y.astype(int)
        mask = np.logical_and(np.logical_and(ar_x_int >= 0, ar_y_int >= 0),
                              np.logical_and(ar_x_int < self.x_count, ar_y_int < self.y_count))
        self.ar_flux[ar_x_int[mask], ar_y_int[mask]] += ar_flux[mask]
        self.ar_count[ar_x_int[mask], ar_y_int[mask]] += 1

    def add_array_with_mask(self, ar_flux, ar_x, ar_y, mask):
        ar_x_int = ar_x.astype('int16')
        ar_y_int = ar_y.astype('int16')
        m = np.logical_and(np.logical_and(np.logical_and(ar_x_int >= 0, ar_y_int >= 0),
                                          np.logical_and(ar_x_int < self.x_count, ar_y_int < self.y_count)),
                           mask)
        ar_flux_new = ar_flux[m]
        ar_indices_x = ar_x_int[m]
        ar_indices_y = ar_y_int[m]
        # make sure we don't invert x an y
        # represent bins in 1D. this is faster than a 2D numpy histogram
        ar_indices_xy = ar_indices_y + (50 * ar_indices_x)
        # bin data according to x,y values
        flux_hist_1d = np.bincount(ar_indices_xy, ar_flux_new, 2500)
        count_hist_1d = np.bincount(ar_indices_xy, minlength=2500)
        # return from 1D to a 2d array
        flux_hist = flux_hist_1d.reshape((50, 50))
        count_hist = count_hist_1d.reshape((50, 50))
        # accumulate new data
        self.ar_flux += flux_hist
        self.ar_count += count_hist

    def add_array_with_mask_preallocate(self, ar_flux, ar_x, ar_y, mask):
        ar_x_int = self.ar_x_int[:ar_x.shape[0], : ar_x.shape[1]]
        ar_x_int[:] = ar_x
        ar_y_int = self.ar_y_int[:ar_y.shape[0], : ar_y.shape[1]]
        ar_y_int[:] = ar_y
        m = np.logical_and(np.logical_and(np.logical_and(ar_x_int >= 0, ar_y_int >= 0),
                                          np.logical_and(ar_x_int < self.x_count, ar_y_int < self.y_count)),
                           mask)
        self.ar_flux[ar_x_int[m], ar_y_int[m]] += ar_flux[m]
        self.ar_count[ar_x_int[m], ar_y_int[m]] += 1

    def merge(self, bins_2d_2):
        self.ar_flux += bins_2d_2.ar_flux
        self.ar_count += bins_2d_2.ar_count

    def save(self, filename):
        np.save(filename, np.dstack((self.ar_flux, self.ar_count)))

    def load(self, filename):
        stacked_array = np.load(filename)
        self.ar_flux = stacked_array[:, :, 0]
        self.ar_count = stacked_array[:, :, 1]
