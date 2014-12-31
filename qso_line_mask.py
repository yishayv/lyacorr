import numpy as np
import spectrum
import collections

#TODO: replace with a more accurate number
lya_center = 1215.67

def rydberg_ratio(m,n):
    return abs(1./(m*m)-1./(n*n))
def ly_series_ratio(n):
    return rydberg_ratio(1,n)
def ly_series_wavelength(n):
    return lya_center*ly_series_ratio(2)/ly_series_ratio(n)


SpectralLine = collections.namedtuple(
    'SpectralLine', ['name', 'wavelength', 'width_factor'])

SpecLines = [
#TODO: replace with a more accurate number
    SpectralLine('Ly-beta', ly_series_wavelength(3),1.03),
    SpectralLine('Ly-alpha', ly_series_wavelength(2), 1.05),
#CIV emission is actually a doublet:
#CIV_line_1 = 1548.202 #f_lu=0.190
#with a weaker line at:
#CIV_line_2 = 1550.772 #f_lu=0.0962
#TODO: figure out if we should do some kind of weighted average
    SpectralLine('CIV', 1548.202, 1.03),
#the rest are from: http://astro.uchicago.edu/~subbarao/newWeb/line.html
#note that their civ line is offset somewhat
    SpectralLine('SiIV_OIV', 1399.8, 1.03),
    SpectralLine('CIII', 1908.27, 1.03),
    SpectralLine('CII', 2326.0, 1.01),#weak
    SpectralLine('MgII', 2800.32, 1.03)]


def is_masked_by_line(wavelength,line_wavelength,line_width_factor,z):
    #print (wavelength<(1+z)*line_wavelength*line_width_factor and
    #        wavelength>(1+z)*line_wavelength/line_width_factor),
    return (wavelength<(1+z)*line_wavelength*line_width_factor and
            wavelength>(1+z)*line_wavelength/line_width_factor)

#vectorize the previous function
vec_is_masked_by_line=np.vectorize(
    is_masked_by_line, excluded=['line_wavelength','line_width_factor','z'])

def create_mask(ar_wavelength,z):
    m = np.zeros(len(ar_wavelength))
    for spec_line in SpecLines:
        current_mask = vec_is_masked_by_line(
            ar_wavelength,spec_line.wavelength, spec_line.width_factor, z)
        print len(current_mask[current_mask>0])
        m = np.logical_or(m, current_mask)
    return m
    
def mask_qso_lines(spec,z):
    m = create_mask(spec.ma_wavelength,z)
    print len(m[m==True])
    spec.ma_wavelength.mask = m
    spec.ma_flux.mask = m
    spec.ma_flux_err.mask = m
    
