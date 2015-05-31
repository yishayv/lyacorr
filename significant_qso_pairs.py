import numpy as np


class SignificantQSOPairs:
    def __init__(self, num_elements=10, dtype=float, initial_value=np.nan):
        self.ar_qso1 = np.zeros(shape=num_elements, dtype=int)
        self.ar_qso2 = np.zeros(shape=num_elements, dtype=int)
        self.ar_values = np.full(shape=num_elements, fill_value=initial_value, dtype=dtype)
        self.current_index_of_minimum = 0

    def add_if_larger(self, qso1, qso2, value):
        # if the current minimum value is smaller than the new value, replace it
        n = self.current_index_of_minimum
        # the condition is negated so that NANs are always replaced.
        if not self.ar_values[n] >= value:
            self.ar_qso1[n] = qso1
            self.ar_qso2[n] = qso2
            self.ar_values[n] = value

            # find the new minimum, and store it for fast access
            self.current_index_of_minimum = self.ar_values.argmin()
