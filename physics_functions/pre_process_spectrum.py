from data_access.qso_data import QSOData
from physics_functions.deredden_func import DereddenSpectrum
from physics_functions.spectrum_calibration import SpectrumCalibration
from physics_functions.remove_mw_lines import MWLines
import common_settings

settings = common_settings.Settings()


class PreProcessSpectrum:
    def __init__(self):
        self.deredden_spectrum = DereddenSpectrum()
        self.spectrum_calibration = SpectrumCalibration(settings.get_tp_correction_hdf5())
        self.mw_lines = MWLines()

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

        new_qso_data = qso_data
        new_qso_data.ar_flux = ar_flux
        new_qso_data.ar_ivar = ar_ivar
        result_string = 'processed'

        return qso_data, result_string
