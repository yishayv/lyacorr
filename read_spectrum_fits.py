import numpy as np
import pyfits
import csv
import itertools
import os.path
import astropy.table as table

QSO_FILE = '../../data/QSOs_test.fit'
# read header names for the QSO table
QSO_FIELDS_FILE = '../../data/QSOs_test_header.csv'
with open(QSO_FIELDS_FILE, mode='rb') as f:
    QSO_fields = list(csv.reader(f))[0]
QSO_fields_dict = dict(zip(QSO_fields, itertools.count()))
PLATE_DIR_DEFAULT = ['/mnt/gastro/sdss/spectro/redux/v5_7_0',
                     '/mnt/gastro/sdss/spectro/redux/v5_7_2']


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
    :rtype : (np.ndarray, np.ndarray, QSORecord)
    @type qso_record_list: list[QSORecord]
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
            header = pyfits.getheader(fits_full_path)
            c0 = header["COEFF0"]
            c1 = header["COEFF1"]
            l = header["NAXIS1"]
            # wavelength grid
            counter = np.arange(0, l)
            ogrid = 10 ** (c0 + c1 * counter)

            # get data
            data = pyfits.getdata(fits_full_path)

        # return requested spectrum
        spec = data[qso_rec.fiberID - 1]

        last_fits_partial_path = fits_partial_path
        yield ogrid, spec, qso_rec


class QSORecord:
    def __init__(self, specObjID, z, ra, dec, plate, mjd, fiberID):
        self.specObjID = specObjID
        self.z = z
        self.ra = ra
        self.dec = dec
        self.plate = plate
        self.mjd = mjd
        self.fiberID = fiberID

    @classmethod
    def from_row(cls, qso_row):
        assert isinstance(qso_row, table.Row)
        return cls(qso_row['specObjID'], qso_row['z'], qso_row['ra'], qso_row['dec'], qso_row['plate'],
                   qso_row['mjd'], qso_row['fiberID'])


    def __str__(self):
        return " ".join([str(self.specObjID), str(self.z), str(self.ra), str(self.dec),
                         str(self.plate), str(self.mjd), str(self.fiberID)])

