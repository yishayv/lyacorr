import itertools
import pprint

import numpy as np
from mpi4py import MPI

import mean_flux
import continuum_fit_pca
from mpi_accumulate import accumulate_over_spectra
import read_spectrum_hdf5
import common_settings
from numpy_spectrum_container import NpSpectrumContainer, NpSpectrumIterator
from lya_data_structures import LyaForestTransmittance
from mpi_helper import l_print_no_barrier
from deredden_func import deredden_spectrum

MAX_WAVELENGTH_COUNT = 4992

comm = MPI.COMM_WORLD

settings = common_settings.Settings()
fit_pca_files = settings.get_pca_continuum_tables()
fit_pca = continuum_fit_pca.ContinuumFitPCA(fit_pca_files[0], fit_pca_files[1], fit_pca_files[2])
z_range = (1.9, 3.5, 0.0001)
stats = {'bad_fit': 0, 'low_continuum': 0, 'low_count': 0, 'empty': 0, 'accepted': 0}


class ContinuumAccumulator:
    def __init__(self, num_spectra):
        self.num_spectra = num_spectra
        self.continuum_fit_file = NpSpectrumContainer(False, self.num_spectra, settings.get_continuum_fit_npy(),
                                                      max_wavelength_count=MAX_WAVELENGTH_COUNT)
        self.n = 0
        # initialize file
        self.continuum_fit_file.zero()

    def accumulate(self, result_enum, ar_qso_indices_list):
        for ar_delta_t, ar_qso_indices in itertools.izip(result_enum, ar_qso_indices_list):
            delta_t = NpSpectrumContainer.from_np_array(ar_delta_t, 1)
            for j, n in itertools.izip(NpSpectrumIterator(delta_t), ar_qso_indices):
                # if self.n >= self.num_spectra:
                # break
                self.continuum_fit_file.set_wavelength(n, j.get_wavelength())
                self.continuum_fit_file.set_flux(n, j.get_flux())
                self.continuum_fit_file.set_ivar(n, j.get_ivar())
                self.n += 1
            l_print_no_barrier("n =", self.n)
        l_print_no_barrier("n =", self.n)
        return self.return_result()

    def return_result(self):
        return self.n


def do_continuum_fit_chunk(qso_record_table):
    start_offset = qso_record_table[0]['index']
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5(),
                                                     table_offset=start_offset)
    num_spectra = len(qso_record_table)
    continuum_chunk = NpSpectrumContainer(False, num_spectra)
    # warning: np.ndarray is not initialized by default. zeroing manually.
    continuum_chunk.zero()
    m = mean_flux.MeanFlux.from_file(settings.get_mean_transmittance_npy())
    # for debugging with a small data set:
    # ignore values with less than 20 sample points
    ar_z_mean_flux, ar_mean_flux = m.get_low_pass_mean(20)
    empty_result = NpSpectrumContainer(False, 0)

    n = 0
    for i in qso_record_table:
        current_qso_data = spectra.return_spectrum(i['index'])
        ar_wavelength = current_qso_data.ar_wavelength
        ar_flux = current_qso_data.ar_flux
        ar_ivar = current_qso_data.ar_ivar
        qso_rec = current_qso_data.qso_rec
        z = qso_rec.z
        assert ar_flux.size == ar_ivar.size

        # extinction correction:
        ar_flux = deredden_spectrum(ar_wavelength, ar_flux, qso_rec.extinction_g)
        # TODO: adjust pipeline variance for extinction

        if not ar_ivar.sum() > 0 or not np.any(np.isfinite(ar_flux)):
            # no useful data
            stats['empty'] += 1
            continue

        fit_spectrum, fit_normalization_factor, is_good_fit = \
            fit_pca.fit(ar_wavelength / (1 + z), ar_flux, ar_ivar, z, boundary_value=np.nan)

        if not is_good_fit:
            stats['bad_fit'] += 1
            l_print_no_barrier("skipped QSO (bad fit): ", qso_rec)
            continue

        continuum_chunk.set_wavelength(n, ar_wavelength)
        continuum_chunk.set_flux(n, fit_spectrum)
        # TODO: find a way to estimate error, or create a file without ivar values.
        continuum_chunk.set_ivar(n, np.ones_like(ar_wavelength))

        stats['accepted'] += 1

        n += 1

    l_print_no_barrier("chunk n =", n, "offset =", start_offset)
    return continuum_chunk


accumulate_over_spectra(do_continuum_fit_chunk, ContinuumAccumulator)
l_print_no_barrier(pprint.pformat(stats))
