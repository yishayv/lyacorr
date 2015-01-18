import numpy as np
import pyfits
import csv
import cProfile
import itertools
import matplotlib.pyplot as plt
import mean_flux

lya_center = 1215.67

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


def return_spectra_2(qso_record_list, plate_dir=PLATE_DIR_DEFAULT):
    """
    function returns a QSO object from the fits files based on the meta_file
    :rtype : (np.ndarray, np.ndarray, QSORecord)
    @type qso_record_list: list[QSORecord]
    """
    for i in qso_record_list:
        # get header
        fits_filename = "%s/%s/spPlate-%s-%s.fits" % \
                        (plate_dir, i.plate, str(i.plate).zfill(4), i.mjd)
        header = pyfits.getheader(fits_filename)
        c0 = header["COEFF0"]
        c1 = header["COEFF1"]
        l = header["NAXIS1"]
        # wavelength grid
        counter = np.arange(0, l)
        ogrid = 10 ** (c0 + c1 * counter)

        # get data
        data = pyfits.getdata(fits_filename)
        spec = data[i.fiberID - 1]

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


def profile_main():
    spec_sample = []
    qso_record_list = []
    for i in itertools.islice(generate_qso_details(), 0, 200):
        rec = QSORecord(i[QSO_fields_dict['specObjID']], i[QSO_fields_dict['z']], i[QSO_fields_dict['ra']],
                        i[QSO_fields_dict['dec']], i[QSO_fields_dict['plate']], i[QSO_fields_dict['mjd']],
                        i[QSO_fields_dict['fiberID']])
        # print rec
        qso_record_list.append(rec)

    spec_sample = return_spectra_2(qso_record_list)

    z_range = (2.1, 3.5, 0.005)
    ar_z_range = np.arange(*z_range)
    m = mean_flux.MeanFlux(*z_range)

    print spec_sample.next()[1]
    for j in spec_sample:
        # TODO: why 3817?
        # TODO: better upper limit
        freq_mask = np.logical_and(j[0] > 3817, j[0] < lya_center * (1 + j[2].z) / 1.05)
        ar_wavelength_clipped = j[0][freq_mask]
        if ar_wavelength_clipped.size < 100:
            print "skipped QSO: ", j[2]
            continue
        print "accepted QSO", j[2]
        ar_flux_clipped = j[1][freq_mask]
        ar_z = ar_wavelength_clipped / lya_center - 1
        ar_flux_binned = np.interp(ar_z_range, ar_z, ar_flux_clipped, left=np.nan, right=np.nan)
        ar_flux_mask = ~np.isnan(ar_flux_binned)
        # TODO: mean flux should be relative transmittance, not absolute flux
        m.add_flux_prebinned(ar_flux_binned, ar_flux_mask)


    plt.plot(ar_z_range, m.get_mean()*100)
    plt.plot(ar_z_range, m.ar_count)
    plt.plot(ar_z_range, m.ar_total_flux)
    plt.show()


cProfile.run('profile_main()', sort=2)
