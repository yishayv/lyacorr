from distutils.core import setup, Extension
import numpy
import sys
import os

# define the extension module
debug = os.getenv('DEBUG', 0) != 0
extra_compile_args = ['-O0'] if debug else ['-O3', '-ffast-math']
bin_pixel_pairs = Extension('bin_pixel_pairs', sources=['bin_pixel_pairs.c'],
                          include_dirs=[numpy.get_include()], extra_compile_args=extra_compile_args)

# run the setup
setup(ext_modules=[bin_pixel_pairs])
