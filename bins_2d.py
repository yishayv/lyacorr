import numpy as np


class Bins2D:
    def __init__(self, x_count, y_count):
        self.ar_flux = np.zeros((x_count, y_count))
        self.ar_count = np.zeros((x_count, y_count))
        self.x_count = x_count
        self.y_count = y_count
        self.index_type = ''
        self.update_index_type()

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
        ar_x_int = ar_x.astype(self.index_type)
        ar_y_int = ar_y.astype(self.index_type)
        m = np.logical_and(np.logical_and(np.logical_and(ar_x_int >= 0, ar_y_int >= 0),
                                          np.logical_and(ar_x_int < self.x_count, ar_y_int < self.y_count)),
                           mask)
        ar_flux_new = ar_flux[m]
        ar_indices_x = ar_x_int[m]
        ar_indices_y = ar_y_int[m]
        # make sure we don't invert x an y
        # represent bins in 1D. this is faster than a 2D numpy histogram
        ar_indices_xy = ar_indices_y + (self.y_count * ar_indices_x)
        # bin data according to x,y values
        flux_hist_1d = np.bincount(ar_indices_xy, ar_flux_new, self.y_count * self.x_count)
        count_hist_1d = np.bincount(ar_indices_xy, minlength=self.y_count * self.x_count)
        # return from 1D to a 2d array
        flux_hist = flux_hist_1d.reshape((self.x_count, self.y_count))
        count_hist = count_hist_1d.reshape((self.x_count, self.y_count))
        # accumulate new data
        self.ar_flux += flux_hist
        self.ar_count += count_hist

    def merge(self, bins_2d_2):
        assert self.ar_flux.shape == bins_2d_2.ar_flux.shape
        assert self.ar_count.shape == bins_2d_2.ar_count.shape
        self.ar_flux += bins_2d_2.ar_flux
        self.ar_count += bins_2d_2.ar_count
        return self

    def save(self, filename):
        np.save(filename, np.dstack((self.ar_flux, self.ar_count)))

    def load(self, filename):
        # TODO: to static
        stacked_array = np.load(filename)
        self.ar_flux = stacked_array[:, :, 0]
        self.ar_count = stacked_array[:, :, 1]
        self.x_count = self.ar_count.shape[0]
        self.y_count = self.ar_count.shape[1]
        self.update_index_type()

    def update_index_type(self):
        # choose integer type according to number of bins
        self.index_type = 'int32' if self.x_count * self.y_count > 32767 else 'int16'

    def __radd__(self, other):
        return self.init_as(self).merge(self).merge(other)

    @classmethod
    def init_as(cls, other):
        """

        :type other: Bins2D
        """
        return cls(other.x_count, other.y_count)

    @classmethod
    def from_other(cls, other):
        new_obj = cls.init_as(other)
        new_obj.merge(other)

    @classmethod
    def from_np_arrays(cls, ar_count, ar_flux):
        """

        :param ar_count: np.array
        :param ar_flux: np.array
        """
        assert ar_count.ndim == ar_flux.ndim == 2
        assert ar_count.shape == ar_flux.shape
        new_bins = cls(ar_count.shape[0], ar_count.shape[1])
        new_bins.ar_count = ar_count
        new_bins.ar_flux = ar_flux
        return new_bins
