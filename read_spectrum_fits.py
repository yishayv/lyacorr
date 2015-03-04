import csv
import itertools
import os.path

import numpy as np
import pyfits
import astropy.table as table

import common_settings
from qso_data import QSORecord, QSOData


settings = common_settings.Settings()

QSO_FILE = settings.get_qso_metadata_fits()
# read header names for the QSO table
QSO_FIELDS_FILE = settings.get_qso_metadata_fields()
with open(QSO_FIELDS_FILE, mode='rb') as f:
    QSO_fields = list(csv.reader(f))[0]
QSO_fields_dict = dict(zip(QSO_fields, itertools.count()))
PLATE_DIR_DEFAULT = settings.get_plate_dir_list()


def generate_qso_details():
    """
    iterate over the QSO table, yielding a dictionary containing the values for each QSO
    """
    data = pyfits.getdata(QSO_FILE)
    for obj in data:
        yield obj


def get_fits_partial_path(qso_rec):
    """
    Returns a relative path for the plate file within a data version directory
    :rtype : basestring
    """
    filename = "spPlate-%s-%s.fits" % \
               (str(qso_rec.plate).zfill(4), qso_rec.mjd)
    return os.path.join(str(qso_rec.plate), filename)


def find_fits_file(plate_dir_list, fits_partial_path):
    """
    Returns a path
    :rtype : basestring
    """
    for plate_dir in plate_dir_list:
        fits_path = os.path.join(plate_dir, fits_partial_path)
        if os.path.exists(fits_path):
            return fits_path
    return None


def return_spectra_2(qso_record_table, plate_dir_list=PLATE_DIR_DEFAULT, pre_sort=True):
    """
    function returns a QSO object from the fits files based on the meta_file
    :type qso_record_table: table.Table
    :rtype : QSOData
    """
    last_fits_partial_path = None
    # sort by plate to avoid reopening files too many times
    if pre_sort:
        qso_record_table.sort(['plate'])

    for i in qso_record_table:
        qso_rec = QSORecord.from_row(i)
        fits_partial_path = get_fits_partial_path(qso_rec)

        # skip reading headers and getting a data object if the filename hasn't changed
        if fits_partial_path != last_fits_partial_path:
            fits_full_path = find_fits_file(plate_dir_list, fits_partial_path)
            if not fits_full_path:
                print "Missing file:", fits_partial_path
                continue


            # get header
            hdu_list = pyfits.open(fits_full_path)
            hdu0_header = hdu_list[0].header
            hdu1_header = hdu_list[1].header

            l1 = hdu1_header["NAXIS1"]

            c0 = hdu0_header["COEFF0"]
            c1 = hdu0_header["COEFF1"]
            l = hdu0_header["NAXIS1"]

            assert l1 == l, "flux and ivar dimensions must be equal"

            # wavelength grid
            counter = np.arange(0, l)
            o_grid = 10 ** (c0 + c1 * counter)

            # get data
            data = hdu_list[0].data
            data1 = hdu_list[1].data

        # return requested spectrum
        spec = data[qso_rec.fiberID - 1]
        ivar = data1[qso_rec.fiberID - 1]

        last_fits_partial_path = fits_partial_path
        yield QSOData(qso_rec, o_grid, spec, ivar)



