import lmfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from astropy.cosmology import Planck13, WMAP5, WMAP7, WMAP9

ar_z = np.arange(0, 5, 0.1)


def delta_dist(params, z1, cosmology, bao_scale):
    return [(cosmology.comoving_distance(z=params['delta_z'].value + z1) -
             cosmology.comoving_distance(z=z1)).value - bao_scale]


def find_bao_redshift(z1, cosmology):
    params = lmfit.Parameters()
    params.add('delta_z', 0.1)
    result = lmfit.minimize(delta_dist, params, kws={'z1': z1, 'cosmology': cosmology, 'bao_scale': 100.})
    delta_z = result.params['delta_z'].value
    return delta_z


print(ar_z)
ar_delta_z_planck13 = [find_bao_redshift(z, Planck13) for z in ar_z]
ar_delta_z_wmap5 = [find_bao_redshift(z, WMAP5) for z in ar_z]
ar_delta_z_wmap7 = [find_bao_redshift(z, WMAP7) for z in ar_z]
ar_delta_z_wmap9 = [find_bao_redshift(z, WMAP9) for z in ar_z]
# print(ar_delta_z_planck13, Planck13.comoving_distance(ar_z + ar_delta_z_planck13) -
#       Planck13.comoving_distance(ar_z))
#
# plt.plot(ar_z, Planck13.comoving_distance(ar_z) / Planck13.comoving_distance(ar_z))
# plt.plot(ar_z, Planck13.comoving_distance(ar_z + ar_delta_z_planck13 - ar_delta_z_wmap7) /
#          Planck13.comoving_distance(ar_z))
# plt.show()

# print(scipy.misc.derivative(func=Planck13.comoving_distance, x0=2, dx=0.1))
# ar_dcmv_dz_planck13 = np.array([scipy.misc.derivative(
#     func=lambda (x): Planck13.comoving_distance(x).value, x0=z, dx=0.01) for z in ar_z])
# ar_dcmv_dz_wmap7 = np.array([scipy.misc.derivative(
#     func=lambda (x): WMAP7.comoving_distance(x).value, x0=z, dx=0.01) for z in ar_z])
# plt.plot(ar_z, -(ar_dcmv_dz_planck13 - ar_dcmv_dz_wmap7) * ar_delta_z_planck13)
# plt.show()
del scipy.misc

ar_base_cmvd_planck13 = Planck13.comoving_distance(ar_z)
ar_true_planck13_cmvd = Planck13.comoving_distance(ar_z + ar_delta_z_planck13)
ar_base_cmvd_wmap5 = WMAP5.comoving_distance(ar_z)
ar_wmap5_apparent_cmvd = WMAP5.comoving_distance(ar_z + ar_delta_z_planck13)
ar_base_cmvd_wmap7 = WMAP7.comoving_distance(ar_z)
ar_wmap7_apparent_cmvd = WMAP7.comoving_distance(ar_z + ar_delta_z_planck13)
ar_base_cmvd_wmap9 = WMAP9.comoving_distance(ar_z)
ar_wmap9_apparent_cmvd = WMAP9.comoving_distance(ar_z + ar_delta_z_planck13)
plt.plot(ar_z, ar_true_planck13_cmvd - ar_base_cmvd_planck13)
plt.plot(ar_z, ar_wmap5_apparent_cmvd - ar_base_cmvd_wmap5)
plt.plot(ar_z, ar_wmap7_apparent_cmvd - ar_base_cmvd_wmap7)
plt.plot(ar_z, ar_wmap9_apparent_cmvd - ar_base_cmvd_wmap9)
# plt.plot(ar_z, ar_wmap7_apparent_cmvd - ar_true_planck13_cmvd)
plt.show()
