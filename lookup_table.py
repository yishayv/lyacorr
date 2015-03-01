import numpy as np


def fast_linear_interpolate(f, x):
    x = np.asarray(x)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1

    # limit the range of x1 to prevent out of bounds access
    return (x1 - x) * f[x0] + (x - x0) * f[np.clip(x1, a_min=0, a_max=f.size - 1)]


class LinearInterpTable:
    def __init__(self, func, x_start, x_end, x_step):
        """

        :type func: a function with a 1D array argument
        :type x_start: float64
        :type x_end: float64
        :type x_step: float64
        """
        self._x_table = np.arange(x_start, x_end, x_step)
        self._func_value_table = func(self._x_table)
        self.x_start = x_start
        self.x_end = x_end
        self.x_step = x_step

    def evaluate(self, ar_x):
        """

        :type ar_x: np.array
        :rtype: float64
        """
        assert np.all(ar_x < self.x_end) & np.all(ar_x > self.x_start), "lookup value out of range"
        ar_index = self._func_value_table.size * (ar_x - self.x_start) / (self.x_end - self.x_start)
        return fast_linear_interpolate(self._func_value_table, ar_index)

