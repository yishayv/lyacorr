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
    SpectralLine('CIV_line', 1548.202, 1.03),
#the rest are from: http://astro.uchicago.edu/~subbarao/newWeb/line.html
#note that their civ line is offset somewhat
    SpectralLine('SiIV_OIV_line', 1399.8, 1.03),
    SpectralLine('CIII_line', 1908.27, 1.03),
    SpectralLine('CII_line', 2326.0, 1.01),#weak
    SpectralLine('MgII_line', 2800.32, 1.03)]


def is_masked_by_line(wavelength,line,width_factor):
    return (wavelength<line/width_factor and
            wavelength>line*width_factor)

def create_mask(ar_wavelength):
    ar_wavelength.where(not is_masked_by_line(lya_center) and
                        not is_masked_by_line(CIV_line) and 
                        not is_masked_by_line(SiIV_OIV_line) and 
                        not is_masked_by_line(CIII_line) and 
                        not is_masked_by_line(MgII_line))
    
def mask_qso_lines(spec):
    spec.ar_wavelength.mask = create_mask(spec.ar_wavelength)
