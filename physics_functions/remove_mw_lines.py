import numpy as np
import pyfits
import healpy as hp

import common_settings

settings = common_settings.Settings()


def load_spectra(path):
    """
    function loads the .fits file of the stacked spectra
    """
    data = pyfits.getdata(path)
    ar_wl = data['wavelength grid']
    # assume that group IDs are integers in the range of [0, num_data_columns-1]
    ar_spectra = np.zeros(shape=(len(data.columns) - 1, ar_wl.size))

    # extract the correct median spectrum
    for c in data.columns:
        if c.name != 'wavelength grid':
            # we expect the column name to contain 'group_id=' followed by a number
            parsed_string = dict([i.split('=') for i in c.name.split(',')])
            # index of group_id in parsed string
            group_id = int(parsed_string['group_id'])
            ar_spectra[group_id] = data[c.name]

    return ar_wl, ar_spectra


class MWLines:
    def __init__(self):
        # convert pixel ID to group ID
        pixel_id, group_id = self.load_metadata(settings.get_mw_pixel_to_group_mapping_fits())
        assert pixel_id.size == group_id.size
        unique_group_id = np.unique(group_id)
        assert np.all(
            unique_group_id == np.append(
                np.arange(unique_group_id.size - 1),
                [10000])), "group IDs must be consecutive integers starting from 0, or 10000"
        assert np.setdiff1d(
            pixel_id, np.arange(pixel_id.size)).size == 0, "pixel IDs must be consecutive integers starting from 0"

        self.fast_group_id = np.zeros_like(group_id, dtype=int)
        # if the pixel_id list is unsorted, assign each pixel's group ID to its pixel offset in fast_group_id.
        self.fast_group_id[pixel_id.astype(int)] = group_id[np.arange(pixel_id.size)]
        # for now load all data to memory on init (~10mb)
        self.ar_wl, self.ar_spectra = load_spectra(settings.get_mw_stacked_spectra_fits())

    def load_metadata(self, path):
        """
        function loads the metadata which contains the connection between pixel ID to group ID
        """
        metadata = pyfits.getdata(path)
        pixel_id = metadata['pixel_id']
        group_id = metadata['Group ID']
        return pixel_id, group_id

    def return_stacked_spectrum_at_coord(self, spec_ra, spec_dec):
        """
        function returns the stacked spectrum that corresponds to the coordinate location that is given
        one should give a RA and DEC coordinates of the needed QSO
        """
        # convert coordinates to pixel ID
        # the ang2pix function can also get a list of theta and phi, no need for iterations
        theta = (-spec_dec + 90) / 180 * np.pi
        phi = (spec_ra % 360) / 180 * np.pi
        pixel = hp.ang2pix(32, theta, phi)

        # convert pixel ID to group ID
        group = self.fast_group_id[pixel]
        if group == 10000:
            print "the coordinates are not in part of the stacked spectra"
            return np.array([]), np.array([]), False

        return self.ar_wl, self.ar_spectra[group], True

    def apply_correction(self, ar_wavelength, ar_flux, ar_ivar, ra, dec):
        ar_wavelength_stacked, ar_flux_stacked, is_corrected = self.return_stacked_spectrum_at_coord(ra, dec)
        if not is_corrected:
            return np.array([]), np.array([]), False

        ar_flux_stacked_resampled = np.interp(ar_wavelength, ar_wavelength_stacked, ar_flux_stacked)

        ar_flux_new = ar_flux / ar_flux_stacked_resampled
        ar_ivar_new = ar_ivar / ar_flux_stacked_resampled ** 2
        return ar_flux_new, ar_ivar_new, True

# mw_lines = MWLines()
# pixel, group, wl, spec = mw_lines.return_stacked_spectrum_at_coord(11.25, 14.1497329363)
# print spec
