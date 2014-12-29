import numpy as np
import scipy as sp
import scipy.optimize as sp_optimize
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
i=413

spectra = np.load('../../data/QSOs_spectra_for_yishay_2.npy')
spectrum = spectra[i]

spec_index = np.genfromtxt('../../data/MyResult_20141225.csv',
                           delimiter=',',
                           skip_header=1)

qso_z = spec_index[i][3]
print qso_z

#create the frequency series for the measurements
freq = np.arange(3817,9206,0.5)

flux=spectra[i]
flux_nonzero=flux[flux>0]
freq_nonzero=freq[flux>0]
print len(flux_nonzero), len(flux)
print len(freq_nonzero), len(freq)

logfreq=np.log(freq_nonzero)
logflux=np.log(flux_nonzero)
yerr=1;

#logflux[(logflux==-np.inf)]=0

# simple power law fit
# define our (line) fitting function
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
pinit = [1.0, 1.0]
out = sp_optimize.leastsq(errfunc, pinit,
args=(logfreq, logflux, yerr), full_output=1)
pfinal = out[0]
covar = out[1]
print pfinal
print covar
index = pfinal[1]
amp = np.e**pfinal[0]
print amp,'*e^',index

indexErr = np.sqrt( covar[0][0] )
ampErr = np.sqrt( covar[1][1] ) * amp

# Define function for calculating a power law
powerlaw = lambda x, amp, index: amp * (x**index)

plt.loglog(freq,flux,'.',linewidth=0.5,ms=2)
#plt.plot(redshift_to_lya_center(qso_z),1,'ro')
plt.axvspan(3817,redshift_to_lya_center(qso_z),
            alpha=0.3,facecolor='yellow',edgecolor='red')
plt.xlim(3e3,1e4);
powerlaw_array=np.vectorize(powerlaw,excluded=['amp','index'])
print powerlaw_array([1,2,3],amp=1,index=2)
plt.loglog(freq,powerlaw_array(freq,amp=amp,index=index))


plt.show()


