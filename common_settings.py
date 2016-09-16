import errno
import os
import os.path
import sys

if sys.version_info < (3,):
    from backports import configparser
else:
    import configparser

_SEP = ':'


class Settings:
    def __init__(self):
        self.config_parser = configparser.ConfigParser()
        self.effective_settings_file_name = os.getenv('LYACORR_CONF_FILE', self.default_settings_file_name)
        if not os.path.exists(self.effective_settings_file_name):
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), self.effective_settings_file_name)
        self.config_parser.read(self.effective_settings_file_name)

    default_settings_file_name = 'lyacorr.rc'

    section_file_paths = 'FilePaths'
    section_performance = 'Performance'
    section_data_processing = 'DataProcessing'
    section_mock_parameters = 'MockParameters'
    section_stacked_ism = 'StackedISM'

    def get_env_expanded_path(self, section, key):
        value = self.config_parser.get(section, key)
        return os.path.expandvars(os.path.expanduser(value))

    def get_env_expanded_option(self, section, key):
        value = self.config_parser.get(section, key)
        return os.path.expandvars(value)

    def get_env_expanded_multiple_paths(self, section, key):
        return [os.path.expanduser(i.strip()) for i in
                self.config_parser.get(section, key).split(_SEP)]

    # File Paths

    def get_plate_dir_list(self):
        """list of paths, separated by comma"""
        return self.get_env_expanded_multiple_paths(self.section_file_paths, 'plate_dir')

    def get_pca_continuum_tables(self):
        """list of 3 required tables (in order)"""
        opt_pca_continuum_tables = 'pca_continuum_tables'
        return self.get_env_expanded_multiple_paths(self.section_file_paths, opt_pca_continuum_tables)

    def get_qso_spectra_hdf5(self):
        """spectra for only QSOs (hdf5 format):"""
        opt_qso_spectra_hdf5 = 'qso_spectra_hdf5'
        return self.get_env_expanded_path(self.section_file_paths, opt_qso_spectra_hdf5)

    def get_mean_transmittance_npy(self):
        """mean transmittance (npy)"""
        opt_mean_transmittance_npy = 'mean_transmittance_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_mean_transmittance_npy)

    def get_median_transmittance_npy(self):
        """median transmittance (npy)"""
        opt_median_transmittance_npy = 'median_transmittance_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_median_transmittance_npy)

    def get_qso_metadata_fits(self):
        """table of QSO metadata (fits)"""
        opt_qso_metadata_fits = 'qso_metadata_fits'
        return self.get_env_expanded_path(self.section_file_paths, opt_qso_metadata_fits)

    def get_qso_metadata_fields(self):
        """fields for the table of QSO metadata (fits)"""
        opt_qso_metadata_fields = 'qso_metadata_fields'
        return self.get_env_expanded_path(self.section_file_paths, opt_qso_metadata_fields)

    def get_qso_metadata_npy(self):
        """table of QSO metadata (npy)"""
        opt_qso_metadata_npy = 'qso_metadata_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_qso_metadata_npy)

    def get_qso_bal_fits(self):
        """table of BAL features (fits)"""
        opt_qso_bal_fits = 'qso_bal_fits'
        return self.get_env_expanded_path(self.section_file_paths, opt_qso_bal_fits)

    def get_delta_t_npy(self):
        """delta_t array (npy)"""
        opt_delta_t_npy = 'delta_transmittance_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_delta_t_npy)

    def get_forest_ism_npy(self):
        """estimated ism component of forest"""
        opt_forest_ism_npy = 'forest_ism_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_forest_ism_npy)

    def get_mean_estimator_bins(self):
        """correlation estimator bins (weighted mean)"""
        opt_mean_estimator_bins = 'mean_estimator_bins_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_mean_estimator_bins)

    def get_median_estimator_bins(self):
        """correlation estimator bins (weighted median)"""
        opt_median_estimator_bins = 'median_estimator_bins_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_median_estimator_bins)

    def get_sigma_squared_lss(self):
        """sigma squared LSS"""
        opt_sigma_sq_lss = 'sigma_squared_lss_txt'
        return self.get_env_expanded_path(self.section_file_paths, opt_sigma_sq_lss)

    def get_weight_eta(self):
        """eta correction function for weights"""
        opt_weight_eta = 'weight_eta_function_txt'
        return self.get_env_expanded_path(self.section_file_paths, opt_weight_eta)

    def get_continuum_fit_npy(self):
        """continuum fit spectra"""
        opt_continuum_fit_npy = 'continuum_fit_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_continuum_fit_npy)

    def get_continuum_fit_metadata_npy(self):
        """continuum fit metadata"""
        opt_continuum_fit_metadata_npy = 'continuum_fit_metadata_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_continuum_fit_metadata_npy)

    def get_fit_snr_stats(self):
        """goodness-of-fit for QSO continua, as a function of signal-to-noise ratio."""
        opt_fit_snr_stats_npy = 'fit_snr_stats_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_fit_snr_stats_npy)

    def get_mean_delta_t_npy(self):
        """mean delta_t per redshift"""
        opt_mean_delta_t_npy = 'mean_delta_t_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_mean_delta_t_npy)

    def get_median_delta_t_npy(self):
        """median delta_t per redshift"""
        opt_median_delta_t_npy = 'median_delta_t_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_median_delta_t_npy)

    def get_significant_qso_pairs_npy(self):
        """list of QSO pairs with most significant contribution to the correlation estimator."""
        opt_significant_qso_pairs_npy = 'significant_qso_pairs_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_significant_qso_pairs_npy)

    def get_tp_correction_hdf5(self):
        """photometric correction to SDSS spectra."""
        opt_tp_correction_hdf5 = 'tp_correction_hdf5'
        return self.get_env_expanded_path(self.section_file_paths, opt_tp_correction_hdf5)

    def get_mw_stacked_spectra_fits(self):
        """stacked spectra for Milky-Way line removal"""
        opt_mw_stacked_spectra_fits = 'mw_stacked_spectra_fits'
        return self.get_env_expanded_path(self.section_file_paths, opt_mw_stacked_spectra_fits)

    def get_mw_pixel_to_group_mapping_fits(self):
        """mapping from pixel ID to group ID for Milky-Way line removal"""
        opt_mw_pixel_to_group_mapping_fits = 'mw_pixel_to_group_mapping_fits'
        return self.get_env_expanded_path(self.section_file_paths, opt_mw_pixel_to_group_mapping_fits)

    def get_ism_extinction_spectra(self):
        """MW lines stacked by extinction"""
        opt_ism_extinction_spectra = 'ism_extinction_spectra_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_ism_extinction_spectra)

    def get_ism_extinction_levels(self):
        """extinction levels for the previous array"""
        opt_ism_extinction_levels = 'ism_extinction_levels_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_ism_extinction_levels)

    def get_correlation_estimator_covariance_npy(self):
        """covariance matrix output"""
        opt_correlation_estimator_covariance_npy = 'correlation_estimator_covariance_npy'
        return self.get_env_expanded_path(self.section_file_paths, opt_correlation_estimator_covariance_npy)

    def get_correlation_estimator_subsamples_npz(self):
        """correlation estimator sub-samples"""
        opt_correlation_estimator_subsamples_npz = 'correlation_estimator_subsamples_npz'
        return self.get_env_expanded_path(self.section_file_paths, opt_correlation_estimator_subsamples_npz)

    # Performance

    def get_file_chunk_size(self):
        """default chunk size for multiprocessing"""
        opt_file_chunk_size = 'file_chunk_size'
        return self.config_parser.getint(self.section_performance, opt_file_chunk_size)

    def get_qso_bundle_size(self):
        """size of QSO bundle to match against all other QSOs."""
        opt_qso_bundle_size = 'qso_bundle_size'
        return self.config_parser.getint(self.section_performance, opt_qso_bundle_size)

    def get_mpi_num_sub_chunks(self):
        """divide MPI tasks to sub-chunks"""
        opt_mpi_num_sub_chunks = 'mpi_num_sub_chunks'
        return self.config_parser.getint(self.section_performance, opt_mpi_num_sub_chunks)

    def get_single_process(self):
        """don't use multiprocessing for easier profiling and debugging"""
        opt_single_process = 'single_process'
        return self.config_parser.getboolean(self.section_performance, opt_single_process)

    def get_profile(self):
        """enable/disable cProfile"""
        opt_profile = 'profile'
        return self.config_parser.getboolean(self.section_performance, opt_profile)

    # Data Processing
    def get_ism_only_mode(self):
        """replace actual forest with estimated ISM"""
        opt_ism_only_mode = 'ism_only_mode'
        return self.config_parser.getboolean(self.section_data_processing, opt_ism_only_mode)

    def get_min_continuum_threshold(self):
        """low continuum flux cutoff"""
        opt_min_continuum_threshold = 'min_continuum_threshold'
        return self.config_parser.getfloat(self.section_data_processing, opt_min_continuum_threshold)

    def get_min_forest_redshift(self):
        """minimum forest redshift to use"""
        opt_min_forest_redshift = 'min_forest_redshift'
        return self.config_parser.getfloat(self.section_data_processing, opt_min_forest_redshift)

    def get_max_forest_redshift(self):
        """maximum forest redshift to use"""
        opt_max_forest_redshift = 'max_forest_redshift'
        return self.config_parser.getfloat(self.section_data_processing, opt_max_forest_redshift)

    def get_num_distance_slices(self):
        """number of distance slices"""
        opt_num_dist_slices = 'num_distance_slices'
        return self.config_parser.getint(self.section_data_processing, opt_num_dist_slices)

    def get_continuum_fit_method(self):
        """continuum fit method"""
        opt_continuum_fit_method = 'continuum_fit_method'
        return self.config_parser.get(self.section_data_processing, opt_continuum_fit_method)

    def get_cosmology(self):
        """cosmology (Planck or WMAP[579])"""
        opt_cosmology = 'cosmology'
        return self.config_parser.get(self.section_data_processing, opt_cosmology)

    def get_healpix_nside(self):
        """healpix nside parameter"""
        opt_healpix_nside = 'healpix_nside'
        return self.config_parser.getint(self.section_data_processing, opt_healpix_nside)

    def get_enable_weighted_mean_estimator(self):
        """enable/disable weighted mean estimator"""
        opt_enable_weighted_mean_estimator = 'enable_weighted_mean_estimator'
        return self.config_parser.getboolean(self.section_data_processing, opt_enable_weighted_mean_estimator)

    def get_enable_weighted_median_estimator(self):
        """enabled/disable weighted median estimator"""
        opt_enable_weighted_median_estimator = 'enable_weighted_median_estimator'
        return self.config_parser.getboolean(self.section_data_processing, opt_enable_weighted_median_estimator)

    def get_enable_mw_line_correction(self):
        """enable MW line correction"""
        opt_enable_mw_line_correction = 'enable_mw_line_correction'
        return self.config_parser.getboolean(self.section_data_processing, opt_enable_mw_line_correction)

    def get_enable_spectrum_flux_correction(self):
        """enable spectrum flux correction"""
        opt_enable_spectrum_flux_correction = 'enable_spectrum_flux_correction'
        return self.config_parser.getboolean(self.section_data_processing, opt_enable_spectrum_flux_correction)

    def get_enable_extinction_correction(self):
        """enable extinction correction"""
        opt_enable_extinction_correction = 'enable_extinction_correction'
        return self.config_parser.getboolean(self.section_data_processing, opt_enable_extinction_correction)

    def get_enable_bal_removal(self):
        """enable bal removal"""
        opt_enable_bal_removal = 'enable_bal_removal'
        return self.config_parser.getboolean(self.section_data_processing, opt_enable_bal_removal)

    def get_enable_estimator_subsamples(self):
        """enable computing the estimator in subsamples, for generating the covariance matrix"""
        opt_enable_estimator_subsamples = 'enable_estimator_subsamples'
        return self.config_parser.getboolean(self.section_data_processing, opt_enable_estimator_subsamples)

    # Mock Parameters

    def get_mock_shell_radius(self):
        """scale of shell in Mpc"""
        opt_mock_shell_radius = 'shell_radius'
        return self.config_parser.getfloat(self.section_mock_parameters, opt_mock_shell_radius)

    def get_mock_fractional_width(self):
        """fractional width of the shell"""
        opt_mock_shell_fractional_width = 'shell_fractional_width'
        return self.config_parser.getfloat(self.section_mock_parameters, opt_mock_shell_fractional_width)

    def get_mock_shell_separation(self):
        """separation from the outermost shell element in Mpc"""
        opt_mock_shell_separation = 'shell_separation'
        return self.config_parser.getfloat(self.section_mock_parameters, opt_mock_shell_separation)

    def get_mock_core_radius(self):
        """core size in Mpc"""
        opt_mock_core_radius = 'core_radius'
        return self.config_parser.getfloat(self.section_mock_parameters, opt_mock_core_radius)

    def get_mock_resolution(self):
        """resolution of the 3d grid"""
        opt_mock_resolution = 'resolution'
        return self.config_parser.getfloat(self.section_mock_parameters, opt_mock_resolution)

    # stacked ISM spectra

    def get_galaxy_metadata_fits(self):
        """galaxy/qso metadata"""
        opt_galaxy_metadata_fits = 'galaxy_metadata_fits'
        return self.get_env_expanded_path(self.section_stacked_ism, opt_galaxy_metadata_fits)

    def get_galaxy_metadata_npy(self):
        """galaxy/qso metadata as an astropy table"""
        opt_galaxy_metadata_npy = 'galaxy_metadata_npy'
        return self.get_env_expanded_path(self.section_stacked_ism, opt_galaxy_metadata_npy)

    def get_ism_histogram_npz(self):
        """histogram output file"""
        opt_ism_histogram_npz = 'ism_histogram_npz'
        return self.get_env_expanded_path(self.section_stacked_ism, opt_ism_histogram_npz)
