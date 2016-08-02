"""
Helper file for loading PCA templates form Suzuki 2005 and Paris 2011.
"""
import numpy as np

import common_settings

settings = common_settings.Settings()


# based on [Paris 2011] and [Lee, Suzuki, & Spergel 2012]
class PCALoaderParis:
    WAVELENGTH_STEP = 0.5
    BLUE_START = 1020
    RED_END = 2000
    RED_END_GOODNESS_OF_FIT = 1600
    LY_A_PEAK_BINNED = 1216
    LY_A_PEAK_INDEX = (LY_A_PEAK_BINNED - BLUE_START) / WAVELENGTH_STEP
    LY_A_NORMALIZATION_BIN = 1280
    LY_A_NORMALIZATION_INDEX = (LY_A_NORMALIZATION_BIN - BLUE_START) / WAVELENGTH_STEP
    NUM_RED_BINS = (RED_END - LY_A_PEAK_BINNED) / WAVELENGTH_STEP + 1
    NUM_FULL_BINS = (RED_END - BLUE_START) / WAVELENGTH_STEP + 1
    RED_END_GOODNESS_OF_FIT_INDEX = RED_END_GOODNESS_OF_FIT / WAVELENGTH_STEP
    PC1_INDEX = 2

    def __init__(self, red_pc_text_file, full_pc_text_file, projection_matrix_file,
                 num_components=8):
        assert 0 < num_components <= 10
        self.red_pc_table = np.genfromtxt(red_pc_text_file, skip_header=0)[:self.NUM_RED_BINS]
        self.full_pc_table = np.genfromtxt(full_pc_text_file, skip_header=0)[:self.NUM_FULL_BINS]
        self.projection_matrix = np.genfromtxt(projection_matrix_file, delimiter=None)[:num_components, :num_components]
        self.num_components = num_components
        self.red_pc = self.red_pc_table[:, self.PC1_INDEX:self.PC1_INDEX + num_components]
        self.full_pc = self.full_pc_table[:, self.PC1_INDEX:self.PC1_INDEX + num_components]
        self.red_mean = self.red_pc_table[:, 1]
        self.full_mean = self.full_pc_table[:, 1]
        # create wavelength bins
        self.ar_wavelength_bins = np.arange(self.BLUE_START, self.RED_END + .1, self.WAVELENGTH_STEP)
        self.ar_red_wavelength_bins = np.arange(self.LY_A_PEAK_BINNED, self.RED_END + .1, self.WAVELENGTH_STEP)
        self.ar_blue_wavelength_bins = np.arange(self.BLUE_START, self.LY_A_PEAK_BINNED, self.WAVELENGTH_STEP)
        # pre-calculated values for mean flux regulation
        self.pivot_wavelength = 1280
        self.delta_wavelength = self.ar_blue_wavelength_bins / self.pivot_wavelength - 1
        self.delta_wavelength_sq = np.square(self.delta_wavelength)


# based on [Suzuki 2005] and [Lee, Suzuki, & Spergel 2012]
class PCALoaderSuzuki:
    WAVELENGTH_STEP = 0.5
    BLUE_START = 1020
    RED_END = 1600
    RED_END_GOODNESS_OF_FIT = 1600
    LY_A_PEAK_BINNED = 1216
    LY_A_PEAK_INDEX = (LY_A_PEAK_BINNED - BLUE_START) / WAVELENGTH_STEP
    LY_A_NORMALIZATION_BIN = 1280
    LY_A_NORMALIZATION_INDEX = (LY_A_NORMALIZATION_BIN - BLUE_START) / WAVELENGTH_STEP
    NUM_RED_BINS = (RED_END - LY_A_PEAK_BINNED) / WAVELENGTH_STEP + 1
    NUM_FULL_BINS = (RED_END - BLUE_START) / WAVELENGTH_STEP + 1
    RED_END_GOODNESS_OF_FIT_INDEX = RED_END_GOODNESS_OF_FIT / WAVELENGTH_STEP
    PC1_INDEX = 3

    def __init__(self, red_pc_text_file, full_pc_text_file, projection_matrix_file,
                 num_components=8):
        assert 0 < num_components <= 10
        self.red_pc_table = np.genfromtxt(red_pc_text_file, skip_header=23)[:self.NUM_RED_BINS]
        self.full_pc_table = np.genfromtxt(full_pc_text_file, skip_header=23)[:self.NUM_FULL_BINS]
        self.projection_matrix = np.genfromtxt(projection_matrix_file, delimiter=',')[:num_components, :num_components]
        self.num_components = num_components
        self.red_pc = self.red_pc_table[:, self.PC1_INDEX:self.PC1_INDEX + num_components]
        self.full_pc = self.full_pc_table[:, self.PC1_INDEX:self.PC1_INDEX + num_components]
        self.red_mean = self.red_pc_table[:, 1]
        self.full_mean = self.full_pc_table[:, 1]
        # create wavelength bins
        self.ar_wavelength_bins = np.arange(self.BLUE_START, self.RED_END + .1, self.WAVELENGTH_STEP)
        self.ar_red_wavelength_bins = np.arange(self.LY_A_PEAK_BINNED, self.RED_END + .1, self.WAVELENGTH_STEP)
        self.ar_blue_wavelength_bins = np.arange(self.BLUE_START, self.LY_A_PEAK_BINNED, self.WAVELENGTH_STEP)
        # pre-calculated values for mean flux regulation
        self.pivot_wavelength = 1280
        self.delta_wavelength = self.ar_blue_wavelength_bins / self.pivot_wavelength - 1
        self.delta_wavelength_sq = np.square(self.delta_wavelength)
