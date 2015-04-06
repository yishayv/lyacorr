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
    opt_plate_dir_list = 'Plate_Dir'
    # list of 3 required tables (in order)
    opt_pca_continuum_tables = 'Pca_Continuum_Tables'
    # spectra for only QSOs (hdf5 format):
    opt_qso_spectra_hdf5 = 'QSO_Spectra_HDF5'
    # mean transmittance (npy)
    opt_mean_transmittance_npy = 'Mean_Transmittance_Npy'
    # table of QSO metadata (fits)
    opt_qso_metadata_fits = 'QSO_Metadata_fits'
    # fields for the table of QSO metadata (fits)
    opt_qso_metadata_fields = 'QSO_Metadata_fields'
    # table of QSO metadata (npy)
    opt_qso_metadata_npy = 'QSO_Metadata_npy'
    # delta_t array (npy)
    opt_delta_t_npy = 'Delta_Transmittance_npy'
    # correlation estimator bins
    opt_estimator_bins = 'Estimator_Bins_npy'
    # sigma squared LSS
    opt_sigma_sq_lss = 'Sigma_Squared_LSS_txt'
    # eta correction function for weights
    opt_weight_eta = 'Weight_Eta_Function_txt'
    # inverse variance of the continuum
    opt_continuum_ivar = 'Continuum_Inverse_Variance_npy'
    # total weight and weighted delta_t values
    opt_total_delta_t = 'Total_Delta_t_npy'

    section_performance = 'Performance'
    # default chunk size for multiprocessing
    opt_file_chunk_size = 'File_Chunk_Size'
    # divide MPI tasks to sub-chunks
    opt_mpi_num_sub_chunks = 'MPI_Num_Sub_Chunks'
    # don't use multiprocessing for easier profiling and debugging
    opt_single_process = 'Single_Process'
    # enable/disable cProfile
    opt_profile = 'Profile'

    section_data_processing = 'DataProcessing'
    # low continuum flux cutoff
    opt_min_continuum_threshold = 'Min_Continuum_Threshold'
    # enable or disable a 2nd-pass mean flux correction
    opt_enable_mean_correction = 'Enable_Mean_Correction'
    # maximum forest redshift to use
    opt_max_forest_redshift = 'Max_Forest_Redshift'

    section_mock_parameters = 'MockParameters'
    # scale of shell in Mpc
    opt_mock_shell_radius = 'Shell_Radius'
    # fractional width of the shell
    opt_mock_shell_fractional_width = 'Shell_Fractional_Width'
    # separation from the outermost shell element in Mpc
    opt_mock_shell_separation = 'Shell_Separation'
    # core size in Mpc
    opt_mock_core_radius = 'Core_Radius'
    # resolution of the 3d grid
    opt_mock_resolution = 'Resolution'

    def write_default_settings(self):
        value_plate_dir_list = _SEP.join(['/mnt/gastro/sdss/spectro/redux/v5_7_0',
                                          '/mnt/gastro/sdss/spectro/redux/v5_7_2'])
        value_pca_continuum_tables = _SEP.join(['../../data/Suzuki/datafile4.txt',
                                                '../../data/Suzuki/datafile3.txt',
                                                '../../data/Suzuki/projection_matrix.csv'])
        value_qso_spectra_hdf5 = '/mnt/gastro/yishay/sdss_QSOs/spectra.hdf5'
        value_mean_transmittance_npy = '../../data/mean_transmittance.npy'
        value_qso_metadata_fits = '../../data/QSOs_test.fit'
        value_qso_metadata_fields = '../../data/QSOs_test_header.csv'
        value_qso_metadata_npy = '../../data/QSO_table.npy'
        value_delta_t_npy = '../../data/delta_t.npy'
        value_estimator_bins_npy = '../../data/estimator_bins.npy'
        value_sigma_sq_lss = '../../data/Sigma_sq_LSS.txt'
        value_weight_eta = '../../data/Weight_eta_func.txt'
        value_continuum_ivar = '../../data/continuum_ivar.npy'
        value_total_delta_t_npy = '../../data/total_delta_t.npy'

        value_file_chunk_size = 10000
        value_mpi_num_sub_chunks = 1440
        value_single_process = False
        value_profile = False

        value_min_continuum_threshold = 0.5
        value_mean_correction = False
        value_max_forest_redshift = 3.2

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
        self.config_parser.set(self.section_file_paths, self.opt_qso_metadata_fits, value_qso_metadata_fits)
        self.config_parser.set(self.section_file_paths, self.opt_qso_metadata_fields, value_qso_metadata_fields)
        self.config_parser.set(self.section_file_paths, self.opt_qso_metadata_npy, value_qso_metadata_npy)
        self.config_parser.set(self.section_file_paths, self.opt_delta_t_npy, value_delta_t_npy)
        self.config_parser.set(self.section_file_paths, self.opt_estimator_bins, value_estimator_bins_npy)
        self.config_parser.set(self.section_file_paths, self.opt_sigma_sq_lss, value_sigma_sq_lss)
        self.config_parser.set(self.section_file_paths, self.opt_weight_eta, value_weight_eta)
        self.config_parser.set(self.section_file_paths, self.opt_continuum_ivar, value_continuum_ivar)
        self.config_parser.set(self.section_file_paths, self.opt_total_delta_t, value_total_delta_t_npy)

        self.config_parser.add_section(self.section_performance)
        self.config_parser.set(self.section_performance, self.opt_file_chunk_size, str(value_file_chunk_size))
        self.config_parser.set(self.section_performance, self.opt_mpi_num_sub_chunks, str(value_mpi_num_sub_chunks))
        self.config_parser.set(self.section_performance, self.opt_single_process, str(value_single_process))
        self.config_parser.set(self.section_performance, self.opt_profile, str(value_profile))

        self.config_parser.add_section(self.section_data_processing)
        self.config_parser.set(self.section_data_processing, self.opt_min_continuum_threshold,
                               str(value_min_continuum_threshold))
        self.config_parser.set(self.section_data_processing, self.opt_enable_mean_correction,
                               bool(value_mean_correction))
        self.config_parser.set(self.section_data_processing, self.opt_max_forest_redshift,
                               float(value_max_forest_redshift))

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

    def get_plate_dir_list(self):
        return self.config_parser.get(self.section_file_paths, self.opt_plate_dir_list).split(_SEP)

    def get_pca_continuum_tables(self):
        return self.config_parser.get(self.section_file_paths, self.opt_pca_continuum_tables).split(_SEP)

    def get_qso_spectra_hdf5(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_spectra_hdf5)

    def get_mean_transmittance_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_mean_transmittance_npy)

    def get_qso_metadata_fits(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_metadata_fits)

    def get_qso_metadata_fields(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_metadata_fields)

    def get_qso_metadata_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_metadata_npy)

    def get_delta_t_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_delta_t_npy)

    def get_estimator_bins(self):
        return self.config_parser.get(self.section_file_paths, self.opt_estimator_bins)

    def get_sigma_squared_lss(self):
        return self.config_parser.get(self.section_file_paths, self.opt_sigma_sq_lss)

    def get_weight_eta(self):
        return self.config_parser.get(self.section_file_paths, self.opt_weight_eta)

    def get_continuum_ivar(self):
        return self.config_parser.get(self.section_file_paths, self.opt_continuum_ivar)

    def get_total_delta_t(self):
        return self.config_parser.get(self.section_file_paths, self.opt_total_delta_t)

    def get_file_chunk_size(self):
        return self.config_parser.getint(self.section_performance, self.opt_file_chunk_size)

    def get_mpi_num_sub_chunks(self):
        return self.config_parser.getint(self.section_performance, self.opt_mpi_num_sub_chunks)

    def get_single_process(self):
        return self.config_parser.getboolean(self.section_performance, self.opt_single_process)

    def get_profile(self):
        return self.config_parser.getboolean(self.section_performance, self.opt_profile)

    def get_min_continuum_threshold(self):
        return self.config_parser.getfloat(self.section_data_processing, self.opt_min_continuum_threshold)

    def get_enable_mean_correction(self):
        return self.config_parser.getboolean(self.section_data_processing, self.opt_enable_mean_correction)

    def get_max_forest_redshift(self):
        return self.config_parser.getfloat((self.section_data_processing, self.opt_max_forest_redshift))

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
