import csv
import itertools
import os.path

import astropy.table as table
import numpy as np
import pyfits

import common_settings
from data_access.qso_data import QSORecord, QSOData
from pixel_flags import PixelFlags, FlagStats

from python_compat import range

settings = common_settings.Settings()  # type: common_settings.Settings

QSO_FILE = settings.get_qso_metadata_fits()
# read header names for the QSO table
QSO_FIELDS_FILE = settings.get_qso_metadata_fields()
with open(QSO_FIELDS_FILE, mode='r') as f:
    QSO_fields = list(csv.reader(f))[0]
QSO_fields_dict = dict(zip(QSO_fields, itertools.count()))
PLATE_DIR_DEFAULT = settings.get_plate_dir_list()

# remove all pixels with AND bits:
AND_MASK = np.bitwise_not(np.uint32(0))
# remove pixels with the following OR bits:
# the first 13 bits do not mask many pixels so we might as well include them.
# the AND mask of 'bright sky' does not always block sky lines.
OR_MASK = PixelFlags.string_to_int(
    'NOPLUG|BADTRACE|BADFLAT|BADARC|MANYBADCOLUMNS|MANYREJECTED|LARGESHIFT|BADSKYFIBER|' +
    'NEARWHOPPER|WHOPPER|SMEARIMAGE|SMEARHIGHSN|SMEARMEDSN|' +
    'BRIGHTSKY')


def generate_qso_details(qso_file=QSO_FILE):
    """
    iterate over the QSO table, yielding a dictionary containing the values for each QSO
    """
    data = pyfits.getdata(qso_file)
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


def enum_spectra(qso_record_table, plate_dir_list=PLATE_DIR_DEFAULT, pre_sort=True, flag_stats=None):
    """
    yields a QSO object from the fits files corresponding to the appropriate qso_record
    :type qso_record_table: table.Table
    :type plate_dir_list: list[string]
    :type pre_sort: bool
    :type flag_stats: FlagStats
    :rtype: list[QSOData]
    """
    last_fits_partial_path = None
    # sort by plate to avoid reopening files too many times
    if pre_sort:
        qso_record_table.sort(['plate'])

    for i in qso_record_table:
        qso_rec = QSORecord.from_row(i)
        fits_partial_path = get_fits_partial_path(qso_rec)

        # skip reading headers and getting data objects if the filename hasn't changed
        if fits_partial_path != last_fits_partial_path:
            fits_full_path = find_fits_file(plate_dir_list, fits_partial_path)
            if not fits_full_path:
                raise Exception("Missing file:", fits_partial_path)

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

            # get flux_data
            flux_data = hdu_list[0].data
            ivar_data = hdu_list[1].data

            and_mask_data = hdu_list[2].data
            or_mask_data = hdu_list[3].data
            last_fits_partial_path = fits_partial_path

        if None in (flux_data, ivar_data, and_mask_data, or_mask_data, o_grid):
            raise Exception("Unexpected uninitialized variables.")
        # return requested spectrum
        ar_flux = flux_data[qso_rec.fiberID - 1]
        ar_ivar = ivar_data[qso_rec.fiberID - 1]
        assert ar_flux.size == ar_ivar.size

        current_and_mask_data = np.asarray(and_mask_data[qso_rec.fiberID - 1])
        current_or_mask_data = np.asarray(or_mask_data[qso_rec.fiberID - 1])
        ar_effective_mask = np.logical_or(current_and_mask_data & AND_MASK,
                                          current_or_mask_data & OR_MASK)

        if flag_stats is not None:
            for bit in range(0, 32):
                flag_stats.flag_count[bit, 0] += (current_and_mask_data & 1).sum()
                flag_stats.flag_count[bit, 1] += (current_or_mask_data & 1).sum()
                current_and_mask_data >>= 1
                current_or_mask_data >>= 1
            flag_stats.pixel_count += current_and_mask_data.size

        # temporary: set ivar to 0 for all bad pixels
        ar_ivar[ar_effective_mask != 0] = 0

        yield QSOData(qso_rec, o_grid, ar_flux, ar_ivar)
