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


class FlagStats:
    FlagNames = \
        {0: 'NOPLUG', 1: 'BADTRACE', 2: 'BADFLAT', 3: 'BADARC',
         4: 'MANYBADCOLUMNS', 5: 'MANYREJECTED', 6: 'LARGESHIFT', 7: 'BADSKYFIBER',
         8: 'NEARWHOPPER', 9: 'WHOPPER', 10: 'SMEARIMAGE', 11: 'SMEARHIGHSN',
         12: 'SMEARMEDSN', 13: 'UNUSED_13', 14: 'UNUSED_14', 15: 'UNUSED_15',
         16: 'NEARBADPIXEL', 17: 'LOWFLAT', 18: 'FULLREJECT', 19: 'PARTIALREJECT',
         20: 'SCATTEREDLIGHT', 21: 'CROSSTALK', 22: 'NOSKY', 23: 'BRIGHTSKY',
         24: 'NODATA', 25: 'COMBINEREJ', 26: 'BADFLUXFACTOR', 27: 'BADSKYCHI',
         28: 'REDMONSTER', 29: 'UNUSED_29', 30: 'UNUSED_30', 31: 'UNUSED_31'}

    def __init__(self):
        self.flag_count = np.zeros(shape=(32, 2), dtype=np.uint64)
        self.pixel_count = np.uint64()

    def bit_fraction(self, bit, and_or):
        return self.flag_count[bit, and_or] / self.pixel_count

    def to_string(self, bit):
        return '{bit_number:4}: {bit_name:24}: AND:{and_fraction:8.2%} OR:{or_fraction:8.2%}'.format(
            bit_number=bit, bit_name=self.FlagNames[bit],
            and_fraction=self.bit_fraction(bit, 0),
            or_fraction=self.bit_fraction(bit, 1))


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


def enum_spectra(qso_record_table, plate_dir_list=PLATE_DIR_DEFAULT, pre_sort=True, flag_stats=None):
    """
    yields a QSO object from the fits files corresponding to the appropriate qso_record
    :type qso_record_table: table.Table
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

            # get flux_data
            flux_data = hdu_list[0].data
            ivar_data = hdu_list[1].data
            # ignore SPPIXMASK bits that tend to block big parts of some spectra
            # (16 NEARBADPIXEL, 17 LOWFLAT, 22 NOSKY, 26 BADFLUXFACTOR).
            and_mask_data = hdu_list[2].data
            or_mask_data = hdu_list[3].data

        # return requested spectrum
        ar_flux = flux_data[qso_rec.fiberID - 1]
        ar_ivar = ivar_data[qso_rec.fiberID - 1]
        assert ar_flux.size == ar_ivar.size
        ar_or_mask = or_mask_data[qso_rec.fiberID - 1] & 0b11111011101111001111111111111111

        if flag_stats.flag_count is not None:
            current_and_mask_data = and_mask_data[qso_rec.fiberID - 1]
            current_or_mask_data = or_mask_data[qso_rec.fiberID - 1]
            for bit in xrange(0, 32):
                flag_stats.flag_count[bit, 0] += (current_and_mask_data & 1).sum()
                flag_stats.flag_count[bit, 1] += (current_or_mask_data & 1).sum()
                current_and_mask_data >>= 1
                current_or_mask_data >>= 1
            flag_stats.pixel_count += current_and_mask_data.size

        # temporary: set ivar to 0 for all bad pixels
        ar_ivar[ar_or_mask != 0] = 0

        last_fits_partial_path = fits_partial_path
        yield QSOData(qso_rec, o_grid, ar_flux, ar_ivar)



