from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [Extension('_matrixnorm', ['_matrixnorm.pyx'], include_dirs = [numpy.get_include()])]

setup(
    name="toolbox",
    version="0.1",
    # build the cython extension
    ext_modules = cythonize(ext_modules),

    # expose these two modules so `import CustomPlot` and `import misc` work
    py_modules=[
        "CustomPlot",
        "misc",
    ],
)
