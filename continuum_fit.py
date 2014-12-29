import numpy as np
import scipy as sp
import scipy.optimize as sp_optimize


def fit_powerlaw(ar_freq, ar_flux, ar_flux_err):
    #remove all zero measurements (since we're looking for a
    #continuum estimate, these points don't contribute to the fit anyway).
    ar_freq_nonzero = ar_freq[ar_flux>0]
    ar_flux_nonzero = ar_flux[ar_flux>0]
    ar_flux_err_nonzero = ar_flux_err[ar_flux>0]
    #compute log-log values
    ar_logfreq = np.log(ar_freq_nonzero)
    ar_logflux = np.log(ar_flux_nonzero)
    ar_logflux_err = ar_flux_err_nonzero / ar_flux_nonzero
    # simple power law fit
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    pinit = [1.0, 1.0]
    out = sp_optimize.leastsq(errfunc, pinit,
    args=(ar_logfreq, ar_logflux, ar_logflux_err), full_output=1)
    pfinal = out[0]
    covar = out[1]
    print pfinal
    print covar
    index = pfinal[1]
    amp = np.e**pfinal[0]
    print amp,'*e^',index
    indexErr = np.sqrt( covar[0][0] )
    ampErr = np.sqrt( covar[1][1] ) * amp
    return amp,index
