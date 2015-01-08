__author__ = 'yishay'

import numpy as np
from astropy import coordinates as coord
from astropy.coordinates import ICRS, Distance
from astropy import units as u
from astropy.coordinates import matching as matching

# x = coord.SkyCoord(ra=10.68458*u.deg, dec=41.26917*u.deg, frame='icrs')

m, n = (280116, 3)
ar_list = np.arange(1, m*n+1).reshape(m, n).T.reshape(m, n) * 30. / m
ar_list[3, 0] = 5
# print ar_list
coord_set = coord.SkyCoord(ra=ar_list[:, 0]*u.degree, dec=ar_list[:, 1]*u.degree,
                   distance=ar_list[:, 2]*u.kpc)
# print coord_set

x = matching.search_around_sky(coord_set, coord_set, .01*u.degree)
# print x, '\n'

# i = np.arange(0, 5)
pairs_with_unity = np.vstack((x[0], x[1]))
pairs = pairs_with_unity.T[pairs_with_unity[1]!=pairs_with_unity[0]]
print pairs.size

# ind_xyz = np.lexsort((ar_list[:, 2], ar_list[:, 1], ar_list[:, 0]))
# print ind_xyz
#
# ind_yxz = np.lexsort((ar_list[:, 2], ar_list[:, 0], ar_list[:, 1]))
# print ind_yxz
#
# ind_zxy = np.lexsort((ar_list[:, 1], ar_list[:, 0], ar_list[:, 2]))
# print ind_zxy
#
# ar_sort_x = ar_list[ind_xyz]
# print ar_sort_x