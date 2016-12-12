"""
Encapsulate multiple stages of spectrum pre-processing.
The individual stages can be enabled/disabled using common_settings.
"""

import numpy as np

import common_settings
from data_access.qso_data import QSOData
from physics_functions.deredden_func import DereddenSpectrum
from physics_functions.remove_bal import RemoveBALSimple
from physics_functions.remove_dla import RemoveDlaByCatalog
from physics_functions.remove_mw_lines import MWLines
from physics_functions.spectrum_calibration import SpectrumCalibration

settings = common_settings.Settings()  # type: common_settings.Settings


class PreProcessSpectrum:
    def __init__(self):
        if settings.get_enable_extinction_correction():
            self.deredden_spectrum = DereddenSpectrum()
        if settings.get_tp_correction_hdf5():
            self.spectrum_calibration = SpectrumCalibration(settings.get_tp_correction_hdf5())
        if settings.get_enable_mw_line_correction():
            self.mw_lines = MWLines()
        if settings.get_enable_bal_removal():
            self.bal = RemoveBALSimple()
        if settings.get_enable_dla_catalog():
            self.dla = RemoveDlaByCatalog()

    def apply(self, qso_data):
        """

        :type qso_data: QSOData
        :return: QSOData
        """

        # flux correction
        if settings.get_enable_spectrum_flux_correction():
            if not self.spectrum_calibration.is_correction_available(qso_data):
                result_string = 'no_flux_calibration'
                return qso_data, result_string

            corrected_qso_data = self.spectrum_calibration.apply_correction(qso_data)
        else:
            corrected_qso_data = qso_data

        ar_wavelength = corrected_qso_data.ar_wavelength
        ar_flux = corrected_qso_data.ar_flux
        ar_ivar = corrected_qso_data.ar_ivar
        qso_rec = corrected_qso_data.qso_rec
        assert ar_flux.size == ar_ivar.size

        # try to correct lines
        if settings.get_enable_mw_line_correction():
            ar_flux, ar_ivar, is_corrected = self.mw_lines.apply_correction(
                ar_wavelength, ar_flux, ar_ivar, qso_rec.ra, qso_rec.dec)
            if not is_corrected:
                result_string = 'no_mw_lines'
                return qso_data, result_string

        # extinction correction:
        if settings.get_enable_extinction_correction():
            ar_flux, ar_ivar = self.deredden_spectrum.apply_correction(
                ar_wavelength, ar_flux, ar_ivar, qso_rec.extinction_g)

        # mask BAL regions
        z_vi = None
        if settings.get_enable_bal_removal():
            bal_mask, z_vi = self.bal.get_mask(qso_rec.plate, qso_rec.mjd, qso_rec.fiberID, ar_wavelength)
            ar_ivar[bal_mask] = 0

        if settings.get_enable_dla_catalog():
            ar_dla_transmittance = self.dla.get_mask(qso_rec.plate, qso_rec.mjd, qso_rec.fiberID, ar_wavelength)
            # clamp low values to 0:
            ar_dla_transmittance[ar_dla_transmittance < 0.8] = 0
            # avoid dividing by zero:
            ar_transmittance_mask = ar_dla_transmittance != 0
            ar_flux[ar_transmittance_mask] /= ar_dla_transmittance[ar_transmittance_mask]
            ar_ivar *= np.square(ar_dla_transmittance)

        new_qso_data = qso_data
        # if we have a visual inspection value for z, use it instead
        if z_vi:
            qso_data.qso_rec.z = z_vi
        new_qso_data.ar_flux = ar_flux
        new_qso_data.ar_ivar = ar_ivar
        result_string = 'processed'

        return qso_data, result_string
