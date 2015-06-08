import ConfigParser
import os.path

_SEP = ':'


class Settings():
    def __init__(self):
        self.config_parser = ConfigParser.SafeConfigParser()
        if not os.path.exists(self.settings_file_name):
            self.write_default_settings()
        self.config_parser.read(self.settings_file_name)

    settings_file_name = 'lyacorr.rc'

    section_file_paths = 'FilePaths'
    # list of paths, separated by comma
    opt_plate_dir_list = 'plate_dir'
    # list of 3 required tables (in order)
    opt_pca_continuum_tables = 'pca_continuum_tables'
    # spectra for only QSOs (hdf5 format):
    opt_qso_spectra_hdf5 = 'qso_spectra_hdf5'
    # mean transmittance (npy)
    opt_mean_transmittance_npy = 'mean_transmittance_npy'
    # median transmittance (npy)
    opt_median_transmittance_npy = 'median_transmittance_npy'
    # table of QSO metadata (fits)
    opt_qso_metadata_fits = 'qso_metadata_fits'
    # fields for the table of QSO metadata (fits)
    opt_qso_metadata_fields = 'qso_metadata_fields'
    # table of QSO metadata (npy)
    opt_qso_metadata_npy = 'qso_metadata_npy'
    # delta_t array (npy)
    opt_delta_t_npy = 'delta_transmittance_npy'
    # correlation estimator bins (weighted mean)
    opt_mean_estimator_bins = 'mean_estimator_bins_npy'
    # correlation estimator bins (weighted median)
    opt_median_estimator_bins = 'median_estimator_bins_npy'
    # sigma squared LSS
    opt_sigma_sq_lss = 'sigma_squared_lss_txt'
    # eta correction function for weights
    opt_weight_eta = 'weight_eta_function_txt'
    # continuum fit spectra
    opt_continuum_fit_npy = 'continuum_fit_npy'
    # continuum fit metadata
    opt_continuum_fit_metadata_npy = 'continuum_fit_metadata_npy'
    # goodness-of-fit for QSO continua, as a function of signal-to-noise ratio.
    opt_fit_snr_stats_npy = 'fit_snr_stats_npy'
    # mean delta_t per redshift
    opt_mean_delta_t_npy = 'mean_delta_t_npy'
    # median delta_t per redshift
    opt_median_delta_t_npy = 'median_delta_t_npy'
    # list of QSO pairs with most significant contribution to the correlation estimator.
    opt_significant_qso_pairs_npy = 'significant_qso_pairs_npy'

    section_performance = 'Performance'
    # default chunk size for multiprocessing
    opt_file_chunk_size = 'file_chunk_size'
    # divide MPI tasks to sub-chunks
    opt_mpi_num_sub_chunks = 'mpi_num_sub_chunks'
    # don't use multiprocessing for easier profiling and debugging
    opt_single_process = 'single_process'
    # enable/disable cProfile
    opt_profile = 'profile'

    section_data_processing = 'DataProcessing'
    # low continuum flux cutoff
    opt_min_continuum_threshold = 'min_continuum_threshold'
    # minimum forest redshift to use
    opt_min_forest_redshift = 'min_forest_redshift'
    # maximum forest redshift to use
    opt_max_forest_redshift = 'max_forest_redshift'
    # continuum fit method
    opt_continuum_fit_method = 'continuum_fit_method'
    # cosmology (Planck or WMAP[579])
    opt_cosmology = 'cosmology'
    # enable/disable weighted mean estimator
    opt_enable_weighted_mean_estimator = 'enable_weighted_mean_estimator'
    # enabled/disable weighted median estimator
    opt_enable_weighted_median_estimator = 'enable_weighted_median_estimator'

    section_mock_parameters = 'MockParameters'
    # scale of shell in Mpc
    opt_mock_shell_radius = 'shell_radius'
    # fractional width of the shell
    opt_mock_shell_fractional_width = 'shell_fractional_width'
    # separation from the outermost shell element in Mpc
    opt_mock_shell_separation = 'shell_separation'
    # core size in Mpc
    opt_mock_core_radius = 'core_radius'
    # resolution of the 3d grid
    opt_mock_resolution = 'resolution'

    def write_default_settings(self):
        value_plate_dir_list = _SEP.join(['/mnt/gastro/sdss/spectro/redux/v5_7_0',
                                          '/mnt/gastro/sdss/spectro/redux/v5_7_2'])
        value_pca_continuum_tables = _SEP.join(['../../data/Suzuki/datafile4.txt',
                                                '../../data/Suzuki/datafile3.txt',
                                                '../../data/Suzuki/projection_matrix.csv'])
        value_qso_spectra_hdf5 = '/mnt/gastro/yishay/sdss_QSOs/spectra.hdf5'
        value_mean_transmittance_npy = '../../data/mean_transmittance.npy'
        value_median_transmittance_npy = '../../data/median_transmittance.npy'
        value_qso_metadata_fits = '../../data/QSOs_test.fit'
        value_qso_metadata_fields = '../../data/QSOs_test_header.csv'
        value_qso_metadata_npy = '../../data/QSO_table.npy'
        value_delta_t_npy = '../../data/delta_t.npy'
        value_mean_estimator_bins_npy = '../../data/mean_estimator_bins.npy'
        value_median_estimator_bins_npy = '../../data/median_estimator_bins.npy'
        value_sigma_sq_lss = '../../data/Sigma_sq_LSS.txt'
        value_weight_eta = '../../data/Weight_eta_func.txt'
        value_continuum_fit_npy = '../../data/continuum_fit.npy'
        value_continuum_fit_metadata_npy = '../../data/continuum_fit_metadata.npy'
        value_fit_snr_stats_npy = '../../data/fit_snr_stats.npy'
        value_mean_delta_t_npy = '../../data/mean_delta_t.npy'
        value_median_delta_t_npy = '../../data/median_delta_t.npy'
        value_significant_qso_pairs_npy = '../../data/significant_qso_pairs.npy'

        value_file_chunk_size = 10000
        value_mpi_num_sub_chunks = 1440
        value_single_process = False
        value_profile = False

        value_min_continuum_threshold = 0.5
        value_min_forest_redshift = 1.96
        value_max_forest_redshift = 3.2
        value_continuum_fit_method = 'dot_product'
        value_cosmology = 'Planck13'
        value_enable_weighted_mean_estimator = True
        value_enable_weighted_median_estimator = True

        value_mock_shell_radius = 150
        value_mock_shell_fractional_width = 0.005
        value_mock_shell_separation = 200
        value_mock_core_radius = 15
        value_mock_resolution = 300

        # replace config parser with an empty one
        self.config_parser = ConfigParser.SafeConfigParser()
        self.config_parser.add_section(self.section_file_paths)
        self.config_parser.set(self.section_file_paths, self.opt_plate_dir_list, value_plate_dir_list)
        self.config_parser.set(self.section_file_paths, self.opt_pca_continuum_tables, value_pca_continuum_tables)
        self.config_parser.set(self.section_file_paths, self.opt_qso_spectra_hdf5, value_qso_spectra_hdf5)
        self.config_parser.set(self.section_file_paths, self.opt_mean_transmittance_npy, value_mean_transmittance_npy)
        self.config_parser.set(self.section_file_paths, self.opt_median_transmittance_npy,
                               value_median_transmittance_npy)
        self.config_parser.set(self.section_file_paths, self.opt_qso_metadata_fits, value_qso_metadata_fits)
        self.config_parser.set(self.section_file_paths, self.opt_qso_metadata_fields, value_qso_metadata_fields)
        self.config_parser.set(self.section_file_paths, self.opt_qso_metadata_npy, value_qso_metadata_npy)
        self.config_parser.set(self.section_file_paths, self.opt_delta_t_npy, value_delta_t_npy)
        self.config_parser.set(self.section_file_paths, self.opt_mean_estimator_bins, value_mean_estimator_bins_npy)
        self.config_parser.set(self.section_file_paths, self.opt_median_estimator_bins, value_median_estimator_bins_npy)
        self.config_parser.set(self.section_file_paths, self.opt_sigma_sq_lss, value_sigma_sq_lss)
        self.config_parser.set(self.section_file_paths, self.opt_weight_eta, value_weight_eta)
        self.config_parser.set(self.section_file_paths, self.opt_continuum_fit_npy, value_continuum_fit_npy)
        self.config_parser.set(self.section_file_paths, self.opt_continuum_fit_metadata_npy,
                               value_continuum_fit_metadata_npy)
        self.config_parser.set(self.section_file_paths, self.opt_fit_snr_stats_npy, value_fit_snr_stats_npy)
        self.config_parser.set(self.section_file_paths, self.opt_mean_delta_t_npy, value_mean_delta_t_npy)
        self.config_parser.set(self.section_file_paths, self.opt_median_delta_t_npy, value_median_delta_t_npy)
        self.config_parser.set(self.section_file_paths, self.opt_significant_qso_pairs_npy,
                               value_significant_qso_pairs_npy)

        self.config_parser.add_section(self.section_performance)
        self.config_parser.set(self.section_performance, self.opt_file_chunk_size, str(value_file_chunk_size))
        self.config_parser.set(self.section_performance, self.opt_mpi_num_sub_chunks, str(value_mpi_num_sub_chunks))
        self.config_parser.set(self.section_performance, self.opt_single_process, str(value_single_process))
        self.config_parser.set(self.section_performance, self.opt_profile, str(value_profile))

        self.config_parser.add_section(self.section_data_processing)
        self.config_parser.set(self.section_data_processing, self.opt_min_continuum_threshold,
                               str(value_min_continuum_threshold))
        self.config_parser.set(self.section_data_processing, self.opt_min_forest_redshift,
                               float(value_min_forest_redshift))
        self.config_parser.set(self.section_data_processing, self.opt_max_forest_redshift,
                               float(value_max_forest_redshift))
        self.config_parser.set(self.section_data_processing, self.opt_continuum_fit_method,
                               str(value_continuum_fit_method))
        self.config_parser.set(self.section_data_processing, self.opt_cosmology,
                               str(value_cosmology))
        self.config_parser.set(self.section_data_processing, self.opt_enable_weighted_mean_estimator,
                               bool(value_enable_weighted_mean_estimator))
        self.config_parser.set(self.section_data_processing, self.opt_enable_weighted_median_estimator,
                               bool(value_enable_weighted_median_estimator))

        self.config_parser.add_section(self.section_mock_parameters)
        self.config_parser.set(self.section_mock_parameters, self.opt_mock_shell_radius, value_mock_shell_radius)
        self.config_parser.set(self.section_mock_parameters, self.opt_mock_shell_fractional_width,
                               value_mock_shell_fractional_width)
        self.config_parser.set(self.section_mock_parameters, self.opt_mock_shell_separation,
                               value_mock_shell_separation)
        self.config_parser.set(self.section_mock_parameters, self.opt_mock_core_radius, value_mock_core_radius)
        self.config_parser.set(self.section_mock_parameters, self.opt_mock_resolution, value_mock_resolution)

        with open(self.settings_file_name, 'wb') as configfile:
            self.config_parser.write(configfile)

    # File Paths

    def get_plate_dir_list(self):
        return self.config_parser.get(self.section_file_paths, self.opt_plate_dir_list).split(_SEP)

    def get_pca_continuum_tables(self):
        return self.config_parser.get(self.section_file_paths, self.opt_pca_continuum_tables).split(_SEP)

    def get_qso_spectra_hdf5(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_spectra_hdf5)

    def get_mean_transmittance_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_mean_transmittance_npy)

    def get_median_transmittance_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_median_transmittance_npy)

    def get_qso_metadata_fits(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_metadata_fits)

    def get_qso_metadata_fields(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_metadata_fields)

    def get_qso_metadata_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_metadata_npy)

    def get_delta_t_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_delta_t_npy)

    def get_mean_estimator_bins(self):
        return self.config_parser.get(self.section_file_paths, self.opt_mean_estimator_bins)

    def get_median_estimator_bins(self):
        return self.config_parser.get(self.section_file_paths, self.opt_median_estimator_bins)

    def get_sigma_squared_lss(self):
        return self.config_parser.get(self.section_file_paths, self.opt_sigma_sq_lss)

    def get_weight_eta(self):
        return self.config_parser.get(self.section_file_paths, self.opt_weight_eta)

    def get_continuum_fit_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_continuum_fit_npy)

    def get_continuum_fit_metadata_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_continuum_fit_metadata_npy)

    def get_fit_snr_stats(self):
        return self.config_parser.get(self.section_file_paths, self.opt_fit_snr_stats_npy)

    def get_mean_delta_t_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_mean_delta_t_npy)

    def get_median_delta_t_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_median_delta_t_npy)

    def get_significant_qso_pairs_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_significant_qso_pairs_npy)

    # Performance

    def get_file_chunk_size(self):
        return self.config_parser.getint(self.section_performance, self.opt_file_chunk_size)

    def get_mpi_num_sub_chunks(self):
        return self.config_parser.getint(self.section_performance, self.opt_mpi_num_sub_chunks)

    def get_single_process(self):
        return self.config_parser.getboolean(self.section_performance, self.opt_single_process)

    def get_profile(self):
        return self.config_parser.getboolean(self.section_performance, self.opt_profile)

    # Data Processing

    def get_min_continuum_threshold(self):
        return self.config_parser.getfloat(self.section_data_processing, self.opt_min_continuum_threshold)

    def get_min_forest_redshift(self):
        return self.config_parser.getfloat(self.section_data_processing, self.opt_min_forest_redshift)

    def get_max_forest_redshift(self):
        return self.config_parser.getfloat(self.section_data_processing, self.opt_max_forest_redshift)

    def get_continuum_fit_method(self):
        return self.config_parser.get(self.section_data_processing, self.opt_continuum_fit_method)

    def get_cosmology(self):
        return self.config_parser.get(self.section_data_processing, self.opt_cosmology)

    def get_enable_weighted_mean_estimator(self):
        return self.config_parser.getboolean(self.section_data_processing, self.opt_enable_weighted_mean_estimator)

    def get_enable_weighted_median_estimator(self):
        return self.config_parser.getboolean(self.section_data_processing, self.opt_enable_weighted_median_estimator)

    # Mock Parameters

    def get_mock_shell_radius(self):
        return self.config_parser.getfloat(self.section_mock_parameters, self.opt_mock_shell_radius)

    def get_mock_fractional_width(self):
        return self.config_parser.getfloat(self.section_mock_parameters, self.opt_mock_shell_fractional_width)

    def get_mock_shell_separation(self):
        return self.config_parser.getfloat(self.section_mock_parameters, self.opt_mock_shell_separation)

    def get_mock_core_radius(self):
        return self.config_parser.getfloat(self.section_mock_parameters, self.opt_mock_core_radius)

    def get_mock_resolution(self):
        return self.config_parser.getfloat(self.section_mock_parameters, self.opt_mock_resolution)
