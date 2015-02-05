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

    section_performance = 'Performance'
    # default chunk size for multiprocessing
    opt_chunk_size = 'Chunk_Size'
    # don't use multiprocessing for easier profiling and debugging
    opt_single_process = 'Single_Process'
    # enable/disable cProfile
    opt_profile = 'Profile'


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

        value_chunk_size = 10000
        value_single_process = False
        value_profile = False

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

        self.config_parser.add_section(self.section_performance)
        self.config_parser.set(self.section_performance, self.opt_chunk_size, str(value_chunk_size))
        self.config_parser.set(self.section_performance, self.opt_single_process, str(value_single_process))
        self.config_parser.set(self.section_performance, self.opt_profile, str(value_profile))

        with open(self.settings_file_name, 'wb') as configfile:
            self.config_parser.write(configfile)

    def get_plate_dir_list(self):
        return self.config_parser.get(self.section_file_paths, self.opt_plate_dir_list).split(_SEP)

    def get_pca_continuum_tables(self):
        return self.config_parser.get(self.section_file_paths, self.opt_pca_continuum_tables).split(_SEP)

    def get_mean_qso_spectra_hdf5(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_spectra_hdf5)

    def get_mean_transmittance_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_mean_transmittance_npy)

    def get_qso_metadata_fits(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_metadata_fits)

    def get_qso_metadata_fields(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_metadata_fields)

    def get_qso_metadata_npy(self):
        return self.config_parser.get(self.section_file_paths, self.opt_qso_metadata_npy)

    def get_chunk_size(self):
        return int(self.config_parser.get(self.section_performance, self.opt_chunk_size))

    def get_single_process(self):
        return bool(self.config_parser.get(self.section_performance, self.opt_single_process))

    def get_profile(self):
        return bool(self.config_parser.get(self.section_performance, self.opt_profile))
