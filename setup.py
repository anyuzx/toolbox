from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


ext_modules = [Extension('_matrixnorm', ['_matrixnorm.pyx'], include_dirs = [numpy.get_include()])]

setup(
  ext_modules = cythonize(ext_modules),
)
