import numpy as np
import matplotlib.pyplot as plt

import continuum_fit

lya_center = 1216.0
def redshift_to_lya_center(z):
    return (1+z)*lya_center

def lya_center_to_redshift(freq):
    return (freq/lya_center)-1

#load spectra from CSV
#spectra = np.genfromtxt('../../data/QSOs_spectra_for_yishay_2.csv',
#                     delimiter=',', skip_footer=736, skip_header=0)

#load a individual spectrum from CSV
count = 740
i=413

spectra = np.load('../../data/QSOs_spectra_for_yishay_2.npy')
spectrum = spectra[i]

spec_index = np.genfromtxt('../../data/MyResult_20141225.csv',
                           delimiter=',',
                           skip_header=1)

qso_z = spec_index[i][3]
print qso_z

#create the frequency series for the measurements
ar_freq = np.arange(3817,9206,0.5)
#use selected spectrum
ar_flux=spectra[i]
#we assume the frequency range in the input file is correct
assert len(ar_freq)==len(ar_flux)

#for now we have no real error data, so just use '1's:
ar_flux_err=np.ones(len(ar_flux));
#mask the Ly-alpha part of the spectrum
#note:if performance becomes a problem, it may be a good idea to remove
#     the masked points completely from the powerlaw fit.
ar_flux_err[ar_freq<redshift_to_lya_center(qso_z)]=np.inf
amp,index = continuum_fit.fit_powerlaw(ar_freq,ar_flux,ar_flux_err)

# Define function for calculating a power law
powerlaw = lambda x, amp, index: amp * (x**index)

plt.loglog(ar_freq,ar_flux,'.',ms=2)
#plt.plot(redshift_to_lya_center(qso_z),1,'ro')
plt.axvspan(3817,redshift_to_lya_center(qso_z),
            alpha=0.3,facecolor='yellow',edgecolor='red')
plt.xlim(3e3,1e4);

#create a predicted flux array, based on fitted powerlaw
powerlaw_array=np.vectorize(powerlaw,excluded=['amp','index'])

ar_flux/powerlaw_array(ar_freq,amp,index)
plt.loglog(ar_freq,ar_flux/powerlaw_array(ar_freq,amp,index),'.',ms=2)
plt.loglog(ar_freq,powerlaw_array(ar_freq,amp=amp,index=index))


plt.show()


