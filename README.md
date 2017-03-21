# lyacorr
This is a set of programs for calculating the autocorrelation of Lyman-α forests from SDSS BOSS data.

## Installation

Deployment is not very user friendly at this point.

### Requirements
- Python 2, or Python 3

#### Required python packages:
- Cython
- matplotlib
- lmfit
- h5py
- healpy
- astropy
- scipy
- numpy
- mpi4py

#### Required data files:
- SDSS BOSS DR12 spPlate fits files, with the original directory structure
- QSO table from a CasJobs query (TODO: add)
- PCA continuum fit tables from [Suzuki et al. 2005][suzuki] and [Paris et al. 2012][paris].
- Broard Absorption Line catalog `DR12Q_BAL.fits`
- DLA catalog from [Garnett et al. 2016][garnett]

[paris]:https://arxiv.org/abs/1104.2024
[suzuki]:http://iopscience.iop.org/article/10.1086/426062/meta
[garnett]:https://arxiv.org/abs/1605.04460
