# lyacorr
This is a set of programs for calculating the autocorrelation of Lyman-α forests from SDSS BOSS data.

## Installation

Note that deployment is not very user friendly.
* Some tinkering with configuration files and scripts may be required.

### Hardware Requirements
- 500GB of storage space
- 32GB of RAM (preferrably more)
- 100GB of fast local storage (RAM or SSD)
- 16 CPU cores or more (recommended)

### Software Requirements
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

#### Cython module
To build and install the cython module run:
```bash
cd cython_helper
python setup.py install --user
```

### Required data files:
- SDSS BOSS DR12 spPlate fits files, with the original directory structure
- QSO table from a CasJobs query, saved as a FITS file, for example:  
  ```SQL
  SELECT S.specObjID,S.mjd,S.plate,S.fiberID,S.ra,S.dec,
         S.zOffset,S.z,S.zErr,S.zWarning,S.class,S.subClass,P.extinction_g
  INTO mydb.QSOs FROM SpecObj AS S
  JOIN SpecPhotoAll AS P ON S.specObjID=P.specObjID
  WHERE S.instrument='BOSS' AND (S.zwarning | 0x10) = 0x10 AND
         S.class='QSO' AND (S.boss_target1 & 0x1FF) = 0 AND
         (S.z>2.1) AND (S.z<3.5) AND (S.plate>3523)
  ```
- PCA continuum fit tables from [Suzuki et al. 2005][suzuki] and [Paris et al. 2011][paris].
- Broard Absorption Line catalog `DR12Q_BAL.fits`
- DLA catalog from [Garnett et al. 2016][garnett]

[paris]:https://arxiv.org/abs/1104.2024
[suzuki]:http://iopscience.iop.org/article/10.1086/426062/meta
[garnett]:https://arxiv.org/abs/1605.04460

### Configuration

There is an annotated configuration file at [example_config/](example_config/).

The default configuration file is `lyacorr.rc` at the current directory.  
The configuration file path can be overridden using the `LYACORR_CONF_FILE` environment variable, e.g.:
```bash
export LYACORR_CONF_FILE=/path/to/config_file.rc
```

## Running

There is an annotated script file at [example_config/](example_config/).

The output files are in native numpy format (`.npy`,`.npz`). 
They can be viewed using matplotlib in a Jupyter notebook.

