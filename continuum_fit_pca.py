__author__ = 'yishay'

import numpy as np

class ContinuumFitPCA:
    def __init__(self, red_pc_text_file, full_pc_text_file, projection_matrix_file):
        self.red_pc_table = np.genfromtxt(red_pc_text_file, skip_header=23)
        self.full_pc_table = np.genfromtxt(full_pc_text_file, skip_header=23)
        self.projection_matrix = np.genfromtxt(projection_matrix_file, delimiter=',')
        self.red_pc = self.red_pc_table[:, 3:13]
        self.full_pc = self.full_pc_table[:, 3:13]
    def red_to_full(self, red_pc_coefficients):
        return np.dot(self.projection_matrix, red_pc_coefficients)
    def project_red_spectrum(self, red_spectrum):
        return np.dot(red_spectrum, self.red_pc)
    def full_spectrum(self, full_pc_coefficients):
        return np.dot(self.full_pc, full_pc_coefficients)