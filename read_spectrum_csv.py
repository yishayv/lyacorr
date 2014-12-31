import numpy as np
import matplotlib.pyplot as plt

import spectrum
import qso_line_mask
import continuum_fit

#TODO: replace with a more accurate number
lya_center = 1215.67

def redshift(wavelength,z):
    return (1+z)*wavelength
def redshift_to_lya_center(z):
    return redshift(lya_center,z)

def lya_center_to_redshift(wavelength):
    return (wavelength/lya_center)-1

def plotvmark(wavelength):
    plt.axvspan(wavelength,wavelength, alpha=0.3, edgecolor='red')


#load spectra from CSV
#spectra = np.genfromtxt('../../data/QSOs_spectra_for_yishay_2.csv',
#                     delimiter=',', skip_footer=736, skip_header=0)

#load a individual spectrum from CSV
count = 740
i=536
#interesting objects: 137, 402, 716, 536(z=3.46, bright!!)
#problematic objects: 0, 712, 715

spectra = np.load('../../data/QSOs_spectra_for_yishay_2.npy')

spec_index = np.genfromtxt('../../data/MyResult_20141225.csv',
                           delimiter=',',
                           skip_header=1)

qso_z = spec_index[i][3]
print qso_z

#create the wavelength series for the measurements
ar_wavelength = np.arange(3817,9206,0.5)
#use selected spectrum
ar_flux=spectra[i]
#we assume the wavelength range in the input file is correct
assert len(ar_wavelength)==len(ar_flux)

#for now we have no real error data, so just use '1's:
ar_flux_err=np.ones(len(ar_flux));

spec = spectrum.Spectrum(ar_flux,ar_flux_err,ar_wavelength)
qso_line_mask.mask_qso_lines(spec,qso_z)
#mask the Ly-alpha part of the spectrum
#note:if performance becomes a problem, it may be a good idea to remove
#     the masked points completely from the powerlaw fit.
#ar_flux_err[ar_wavelength<redshift_to_lya_center(qso_z)*1.06]=np.inf
amp,index = continuum_fit.fit_powerlaw(
    spec.ma_wavelength.compressed(),
    spec.ma_flux.compressed(),
    spec.ma_flux_err.compressed())

# Define function for calculating a power law
powerlaw = lambda x, amp, index: amp * (x**index)

plt.loglog(ar_wavelength,ar_flux,'.',ms=2)
plt.axvspan(3817,redshift_to_lya_center(qso_z),
            alpha=0.3,facecolor='yellow',edgecolor='red')

for l in qso_line_mask.SpecLines:
    plotvmark(redshift(l.wavelength,qso_z))
    plt.axvspan(redshift(l.wavelength/l.width_factor,qso_z),
                redshift(l.wavelength*l.width_factor,qso_z),
                alpha=0.2,facecolor='cyan',edgecolor='none')

plt.xlim(3e3,1e4);

#create a predicted flux array, based on fitted powerlaw
powerlaw_array=np.vectorize(powerlaw,excluded=['amp','index'])

ar_flux/powerlaw_array(ar_wavelength,amp,index)
plt.loglog(ar_wavelength,
           ar_flux/powerlaw_array(ar_wavelength,amp,index),'.',ms=2)
plt.loglog(ar_wavelength,
           powerlaw_array(ar_wavelength,amp=amp,index=index))


plt.show()


