import itertools
import pprint

import numpy as np
from mpi4py import MPI

import mean_flux
from continuum_fit_pca import ContinuumFitPCA, ContinuumFitContainer, ContinuumFitContainerFiles
from mpi_accumulate import accumulate_over_spectra
import read_spectrum_hdf5
import common_settings
from numpy_spectrum_container import NpSpectrumContainer, NpSpectrumIterator
from mpi_helper import l_print_no_barrier
from deredden_func import deredden_spectrum


MAX_WAVELENGTH_COUNT = 4992

comm = MPI.COMM_WORLD

settings = common_settings.Settings()
fit_pca_files = settings.get_pca_continuum_tables()
fit_pca = ContinuumFitPCA(fit_pca_files[0], fit_pca_files[1], fit_pca_files[2])
z_range = (1.9, 3.5, 0.0001)
stats = {'bad_fit': 0, 'low_continuum': 0, 'low_count': 0, 'empty': 0, 'accepted': 0}


class ContinuumAccumulator:
    def __init__(self, num_spectra):
        self.num_spectra = num_spectra
        self.continuum_fit_container = ContinuumFitContainerFiles(
            create_new=True, num_spectra=self.num_spectra)
        self.n = 0

    def accumulate(self, result_enum, ar_qso_indices_list, object_all_results):
        for ar_continua, ar_qso_indices, object_result in itertools.izip(
                result_enum, ar_qso_indices_list, object_all_results):

            continua = ContinuumFitContainer.from_np_array_and_object(ar_continua, object_result)
            for i in xrange(continua.num_spectra):
                n = ar_qso_indices[i]
                self.continuum_fit_container.set_wavelength(n, continua.get_wavelength(i))
                self.continuum_fit_container.set_flux(n, continua.get_flux(i))
                # TODO: refactor
                self.continuum_fit_container.copy_metadata(n, continua.get_metadata(i))
                self.n += 1
            l_print_no_barrier("n =", self.n)
        l_print_no_barrier("n =", self.n)
        return self.return_result()

    def return_result(self):
        return self.n

    def finalize(self):
        self.continuum_fit_container.save()


def do_continuum_fit_chunk(qso_record_table):
    start_offset = qso_record_table[0]['index']
    spectra = read_spectrum_hdf5.SpectraWithMetadata(qso_record_table, settings.get_qso_spectra_hdf5())
    num_spectra = len(qso_record_table)
    continuum_chunk = ContinuumFitContainer(num_spectra)

    m = mean_flux.MeanFlux.from_file(settings.get_mean_transmittance_npy())
    # for debugging with a small data set:
    # ignore values with less than 20 sample points
    ar_z_mean_flux, ar_mean_flux = m.get_low_pass_mean(20)

    for n in xrange(len(qso_record_table)):
        current_qso_data = spectra.return_spectrum(n)
        current_qso_index = current_qso_data.qso_rec.index
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

        stats['accepted'] += 1

    l_print_no_barrier("offset =", start_offset)
    return continuum_chunk


accumulate_over_spectra(do_continuum_fit_chunk, ContinuumAccumulator)
l_print_no_barrier(pprint.pformat(stats))
