import numpy as np
import pyfits
import csv
import cProfile
import itertools

QSO_file = '../../data/QSOs.fit'
# read header names for the QSO table
QSO_fields_file = '../../data/QSOs_header.csv'
with open(QSO_fields_file, mode='rb') as f:
    QSO_fields = list(csv.reader(f))[0]
QSO_fields_dict = dict(zip(QSO_fields, itertools.count()))


def generate_QSO_details():
    """
    iterate over the QSO table, yielding a dictionary containing the values for each QSO
    """
    data = pyfits.getdata(QSO_file)
    for obj in data:
        # yield dict(zip(QSO_fields, [obj[0],obj[1]]))
        # yield obj[QSO_fields_dict['z']]
        # yield dict(itertools.izip(QSO_fields, obj))
        yield obj

def return_spectra(meta_file, plate_dir='/mnt/gastro/sdss/spectro/redux/v5_7_0'):
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


def profile_main():
    for j in itertools.islice(generate_QSO_details(), 0, 100):
        print j[QSO_fields_dict['specObjID']], j[QSO_fields_dict['z']]

cProfile.run('profile_main()', sort=2)