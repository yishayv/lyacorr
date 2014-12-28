import numpy as np
import matplotlib.pyplot as plt

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
i=633

spectra = np.load('../../data/QSOs_spectra_for_yishay_2.npy')
spectrum = spectra[i]

spec_index = np.genfromtxt('../../data/MyResult_20141225.csv',
                           delimiter=',',
                           skip_header=1)

qso_z = spec_index[i][3]
print qso_z

#create the frequency series for the measurements
freq = np.arange(3817,9206,0.5)

logfreq=np.log(freq)
#spectrum=spectra[1]
logflux=np.log(spectrum)

plt.loglog(freq,logflux,'.',linewidth=0.5,ms=2)
#plt.plot(redshift_to_lya_center(qso_z),1,'ro')
plt.axvspan(3817,redshift_to_lya_center(qso_z),
            alpha=0.3,facecolor='yellow',edgecolor='red')
plt.xlim(3e3,1e4);
plt.show()


