import numpy as np


class DeltaFSNRBins(object):
    NUM_SNR_BINS = 50
    NUM_DELTA_F_BINS = 50
    LOG_SNR_RANGE = 5.
    LOG_SNR_OFFSET = 1.6
    DELTA_F_RANGE = 1.
    DELTA_F_OFFSET = 0.

    def __init__(self):
        pass

    def snr_to_bin(self, snr):
        if snr <= 0:
            return 0
        return self.log_snr_to_bin(np.log(snr))

    def log_snr_to_bin(self, log_snr):
        return int(np.clip((log_snr + self.LOG_SNR_OFFSET) * self.NUM_SNR_BINS / self.LOG_SNR_RANGE,
                           0, self.NUM_SNR_BINS - 1))

    def bin_to_log_snr(self, bin_num):
        return bin_num * self.LOG_SNR_RANGE / self.NUM_SNR_BINS - self.LOG_SNR_OFFSET

    def delta_f_to_bin(self, delta_f):
        return int(np.clip((delta_f + self.DELTA_F_OFFSET) * self.NUM_DELTA_F_BINS / self.DELTA_F_RANGE,
                           0, self.NUM_DELTA_F_BINS - 1))

    def bin_to_delta_f(self, bin_num):
        return bin_num * self.DELTA_F_RANGE / self.NUM_DELTA_F_BINS - self.DELTA_F_OFFSET

    def get_empty_histogram_array(self):
        return np.zeros(shape=(3, self.NUM_SNR_BINS, self.NUM_DELTA_F_BINS))

    def get_log_snr_axis(self):
        return self.bin_to_log_snr(np.arange(self.NUM_SNR_BINS))

    def get_delta_f_axis(self):
        return self.bin_to_delta_f(np.arange(self.NUM_DELTA_F_BINS))
