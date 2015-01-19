import numpy as np
import pyfits
import csv
import itertools

QSO_FILE = '../../data/QSOs_test.fit'
# read header names for the QSO table
QSO_FIELDS_FILE = '../../data/QSOs_test_header.csv'
with open(QSO_FIELDS_FILE, mode='rb') as f:
    QSO_fields = list(csv.reader(f))[0]
QSO_fields_dict = dict(zip(QSO_fields, itertools.count()))
PLATE_DIR_DEFAULT = '/mnt/gastro/sdss/spectro/redux/v5_7_0'


def generate_qso_details():
    """
    iterate over the QSO table, yielding a dictionary containing the values for each QSO
    """
    data = pyfits.getdata(QSO_FILE)
    for obj in data:
        # yield dict(zip(QSO_fields, [obj[0],obj[1]]))
        # yield obj[QSO_fields_dict['z']]
        # yield dict(itertools.izip(QSO_fields, obj))
        yield obj


def return_spectra(meta_file, plate_dir=PLATE_DIR_DEFAULT):
    """
    function returns a QSO object from the fits files based on the meta_file
    """
    # get a list of plates, mjds and fibers
    plate_list, mjd_list, fiber_list = np.loadtxt(meta_file, delimiter=",", skiprows=1, usecols=[0, 1, 2],
                                                  unpack=True)

    wl_list = []
    spec_list = []
    for i in xrange(len(plate_list)):
        plate_val = int(plate_list[i])
        mjd_val = int(mjd_list[i])
        fiber_val = int(fiber_list[i])

        # get header
        fits_filename = "%s/%s/spPlate-%s-%s.fits" % \
                        (plate_dir, plate_val, str(plate_val).zfill(4), mjd_val)
        header = pyfits.getheader(fits_filename)
        c0 = header["COEFF0"]
        c1 = header["COEFF1"]
        l = header["NAXIS1"]
        # wavelength grid
        counter = np.arange(0, l)
        ogrid = 10 ** (c0 + c1 * counter)

        # get data
        data = pyfits.getdata(fits_filename)
        spec = data[fiber_val - 1]

        # interpolate grid
        # fill if you want
        wl_list.append(ogrid)
        spec_list.append(spec)

        yield ogrid, spec


def return_spectra_2(qso_record_list, plate_dir=PLATE_DIR_DEFAULT, pre_sort=True):
    """
    function returns a QSO object from the fits files based on the meta_file
    :rtype : (np.ndarray, np.ndarray, QSORecord)
    @type qso_record_list: list[QSORecord]
    """
    last_fits_filename = None
    # sort by plate
    if pre_sort:
        qso_record_list_internal = sorted(qso_record_list, key=lambda x: x.plate)

    for i in qso_record_list_internal:
        # get header
        fits_filename = "%s/%s/spPlate-%s-%s.fits" % \
                        (plate_dir, i.plate, str(i.plate).zfill(4), i.mjd)

        # skip reading headers and getting a data object if the filename hasn't changed
        if fits_filename != last_fits_filename:
            header = pyfits.getheader(fits_filename)
            c0 = header["COEFF0"]
            c1 = header["COEFF1"]
            l = header["NAXIS1"]
            # wavelength grid
            counter = np.arange(0, l)
            ogrid = 10 ** (c0 + c1 * counter)

            # get data
            data = pyfits.getdata(fits_filename)

        # return requested spectrum
        spec = data[i.fiberID - 1]

        last_fits_filename = fits_filename
        yield ogrid, spec, i


class QSORecord:
    def __init__(self, specObjID, z, ra, dec, plate, mjd, fiberID):
        self.specObjID = specObjID
        self.z = z
        self.ra = ra
        self.dec = dec
        self.plate = plate
        self.mjd = mjd
        self.fiberID = fiberID

    def __str__(self):
        return " ".join([str(self.specObjID), str(self.z), str(self.ra), str(self.dec),
                         str(self.plate), str(self.mjd), str(self.fiberID)])

