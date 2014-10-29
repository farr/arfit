from Cython.Build import cythonize
from distutils.core import setup
import numpy as np

setup(ext_modules = cythonize('arfit/arn_loops.pyx'),
      include_dirs = [np.get_include()])
      
