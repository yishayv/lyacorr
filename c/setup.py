from distutils.core import setup, Extension
import numpy

# define the extension module
bin_pixel_pairs = Extension('bin_pixel_pairs', sources=['bin_pixel_pairs.c'],
                          include_dirs=[numpy.get_include()], extra_compile_args=['-Ofast'])

# run the setup
setup(ext_modules=[bin_pixel_pairs])
