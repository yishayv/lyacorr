#!/usr/bin/env python

"""
setup.py  to build bin_pixel_pairs code with cython
"""
from distutils.core import setup
from distutils.extension import Extension

import numpy  # to get includes
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("bin_pairs_cython", ["bin_pixel_pairs.pyx"], )],
    include_dirs=[numpy.get_include(), ],
)
