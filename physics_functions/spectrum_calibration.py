import h5py
from scipy.interpolate import interp1d

from data_access.qso_data import QSOData


class SpectrumCalibration:
    """
    This class uses data and code from (Margala et al. 2015, http://arxiv.org/pdf/1506.04790v1.pdf) to
    perform spectrophotometric correction to SDSS spectra.
    """
    def __init__(self, tpcorr_file):
        # Open the throughput correction file
        self.tpcorr = h5py.File(tpcorr_file, 'r')
        self.tpcorr_wave = self.tpcorr['wave'].value

    def apply_correction(self, qso_data):
        """

        :type qso_data: QSOData
        :rtype QSOData
        """
        qso_rec = qso_data.qso_rec
        # Read the target's throughput correction vector
        tpcorr_key = '%s/%s/%s' % (qso_rec.plate, qso_rec.mjd, qso_rec.fiberID)
        if tpcorr_key in self.tpcorr:
            correction = self.tpcorr[tpcorr_key].value

            # Create an interpolated correction function
            correction_interp = interp1d(self.tpcorr_wave, correction, kind='linear')

            # Sample the interpolated correction using the observation's wavelength grid
            resampled_correction = correction_interp(qso_data.ar_wavelength)

            # Apply the correction to the observed flux and ivar
            corrected_flux = qso_data.ar_flux * resampled_correction
            corrected_ivar = qso_data.ar_ivar / resampled_correction ** 2

            # return a new QSOData object with the corrected spectrum
            new_qso_data = QSOData(qso_rec, qso_data.ar_wavelength, corrected_flux, corrected_ivar)
            return new_qso_data
        else:
            print "No flux correction for QSO:", qso_rec
            return qso_data

    def is_correction_avaliable(self, qso_data):
        """

        :type qso_data: QSOData
        :rtype QSOData
        """
        qso_rec = qso_data.qso_rec
        # Read the target's throughput correction vector
        tpcorr_key = '%s/%s/%s' % (qso_rec.plate, qso_rec.mjd, qso_rec.fiberID)
        return tpcorr_key in self.tpcorr