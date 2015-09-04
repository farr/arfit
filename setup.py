from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np

setup(name='ARFit',
      version='0.1',
      description='Code for fitting (C)AR(MA) processes.',
      author='Will M. Farr',
      author_email='w.farr@bham.ac.uk',
      packages=['arfit'],
      scripts=['arfit/run_carma_pack.py',
               'arfit/run_carma_pack_posterior.py',
               'arfit/carma_pack_postprocess.py'],
      ext_modules = cythonize([Extension('arn_loops', ['arfit/arn_loops.pyx']),
                               Extension('ar1_kalman_core', ['arfit/ar1_kalman_core.pyx'])]),
      include_dirs = [np.get_include()])
      
