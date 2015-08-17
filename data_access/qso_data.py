import astropy.table as table


class QSORecord:
    def __init__(self, index, spec_obj_id, z, ra, dec, plate, mjd, fiber_id, extinction_g):
        self.index = index
        self.specObjID = spec_obj_id
        self.z = z
        self.ra = ra
        self.dec = dec
        self.plate = plate
        self.mjd = mjd
        self.fiberID = fiber_id
        self.extinction_g = extinction_g

    @classmethod
    def from_row(cls, qso_row):
        assert isinstance(qso_row, table.Row)
        return cls(qso_row['index'], qso_row['specObjID'], qso_row['z'],
                   qso_row['ra'], qso_row['dec'],
                   qso_row['plate'], qso_row['mjd'], qso_row['fiberID'],
                   qso_row['extinction_g'])


    def __str__(self):
        return " ".join([str(self.specObjID), str(self.z), str(self.ra), str(self.dec),
                         str(self.plate), str(self.mjd), str(self.fiberID)])


class QSOData:
    def __init__(self, qso_rec, ar_wavelength, ar_flux, ar_ivar):
        """

        :type qso_rec: QSORecord
        :type ar_wavelength: np.multiarray.ndarray
        :type ar_flux: np.multiarray.ndarray
        :type ar_ivar: np.multiarray.ndarray
        """
        self.qso_rec = qso_rec
        self.ar_wavelength = ar_wavelength
        self.ar_flux = ar_flux
        self.ar_ivar = ar_ivar
