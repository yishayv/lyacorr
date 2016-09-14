"""
    This module is responsible for calculating the ism transmittance.
    It uses MPI to select different spectra for different nodes, and then combines
    the results in Rank 0 (root).
    The spectra are processed in chunks, and gathered to the root rank after each sub-chunk.
"""
import pprint
from collections import Counter

import numpy as np

import common_settings
import data_access.numpy_spectrum_container
from data_access.numpy_spectrum_container import NpSpectrumContainer, NpSpectrumIterator
from data_access.read_spectrum_fits import QSORecord
from mpi_accumulate import accumulate_over_spectra, comm
from mpi_helper import l_print_no_barrier
from physics_functions.pre_process_spectrum import PreProcessSpectrum
from python_compat import range, zip

lya_center = 1215.67

settings = common_settings.Settings()
force_single_process = settings.get_single_process()
z_range = (1.9, 3.5, 0.0004)
ar_z_range = np.arange(*z_range)
min_continuum_threshold = settings.get_min_continuum_threshold()
local_stats = Counter({'bad_fit': 0, 'low_continuum': 0, 'low_count': 0, 'empty': 0, 'accepted': 0})
pre_process_spectrum = PreProcessSpectrum()
ar_extinction_levels = np.load(settings.get_ism_extinction_levels())

ism_spectra = data_access.numpy_spectrum_container.NpSpectrumContainer(
    readonly=True, create_new=False, filename=settings.get_ism_extinction_spectra(),
    max_wavelength_count=10880)


class ISMTransmittanceAccumulator:
    """
        Modify existing delta transmittance file.
        Replace forest with ISM spectra.
        It is intended to be used as a helper object called by mpi_accumulate.accumulate_over_spectra
    """

    def __init__(self, num_spectra):
        self.num_spectra = num_spectra
        self.forest_ism_file = NpSpectrumContainer(False, num_spectra=self.num_spectra,
                                                   filename=settings.get_forest_ism_npy(), max_wavelength_count=1000)
        self.n = 0
        # initialize file
        self.forest_ism_file.zero()

    def accumulate(self, result_enum, ar_qso_indices_list, object_results):
        # unused parameter:
        del object_results
        for ar_chunk, ar_qso_indices in zip(result_enum, ar_qso_indices_list):
            forest_chunk = NpSpectrumContainer.from_np_array(ar_chunk, readonly=True)
            for j, n in zip(NpSpectrumIterator(forest_chunk), ar_qso_indices):
                # if self.n >= self.num_spectra:
                # break
                self.forest_ism_file.set_wavelength(n, j.get_wavelength())
                self.forest_ism_file.set_flux(n, j.get_flux())
                self.forest_ism_file.set_ivar(n, j.get_ivar())
                self.n += 1
            l_print_no_barrier("n =", self.n)
        l_print_no_barrier("n =", self.n)
        return self.return_result()

    def return_result(self):
        return self.n, None

    def finalize(self):
        pass


def ism_transmittance_chunk(qso_record_table):
    start_offset = qso_record_table[0]['index']
    # spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5())
    # continuum_fit_file = NpSpectrumContainer(True, filename=settings.get_continuum_fit_npy())
    delta_transmittance_file = NpSpectrumContainer(readonly=True,
                                                   filename=settings.get_delta_t_npy(), max_wavelength_count=1000)

    num_spectra = len(qso_record_table)
    ism_delta_t = NpSpectrumContainer(False, num_spectra=num_spectra)
    # warning: np.ndarray is not initialized by default. zeroing manually.
    ism_delta_t.zero()
    n = 0
    for i in range(len(qso_record_table)):
        qso_rec = QSORecord.from_row(qso_record_table[i])
        index = qso_rec.index

        # read original delta transmittance
        ar_redshift = delta_transmittance_file.get_wavelength(index)
        # ar_flux = delta_transmittance_file.get_flux(index)
        ar_ivar = delta_transmittance_file.get_ivar(index)

        # get correction to ISM
        # ar_flux_new, ar_ivar_new, is_corrected = pre_process_spectrum.mw_lines.apply_correction(
        #     ar_wavelength, np.ones_like(ar_flux), ar_ivar, qso_rec.ra, qso_rec.dec)

        ar_wavelength = (ar_redshift + 1) * lya_center  # type: np.ndarray
        # limit maximum bin number because higher extinction bins are not reliable
        max_extinction_bin = max(20, ar_extinction_levels.size)

        if np.isfinite(qso_rec.extinction_g):
            extinction_bin = int(
                np.round(np.interp(qso_rec.extinction_g, ar_extinction_levels, np.arange(max_extinction_bin))))
        else:
            extinction_bin = 0

        l_print_no_barrier("extinction_bin = ", extinction_bin)
        ar_ism_resampled = np.interp(ar_wavelength,
                                     ism_spectra.get_wavelength(extinction_bin),
                                     ism_spectra.get_flux(extinction_bin), left=np.nan, right=np.nan)
        extinction = ar_extinction_levels[extinction_bin]
        # rescale according to QSO extinction
        l_print_no_barrier(qso_rec.extinction_g, extinction)
        ism_scale_factor = 1.
        ar_flux_new = (ar_ism_resampled - 1) * ism_scale_factor * qso_rec.extinction_g / extinction

        ism_delta_t.set_wavelength(i, ar_redshift)
        # use reciprocal to get absorption spectrum, then subtract 1 to get the delta
        ism_delta_t.set_flux(i, ar_flux_new)
        # ism_delta_t.set_flux(i, np.ones_like(ar_flux) * qso_rec.extinction_g)
        # use original ivar because we are not correcting an existing spectrum
        ism_delta_t.set_ivar(i, ar_ivar)

        n += 1

    l_print_no_barrier("chunk n =", n, "offset =", start_offset)
    return ism_delta_t.as_np_array(), None


def calc_ism_transmittance():
    comm.Barrier()
    accumulate_over_spectra(ism_transmittance_chunk,
                            ISMTransmittanceAccumulator)
    l_print_no_barrier(pprint.pformat(local_stats))
