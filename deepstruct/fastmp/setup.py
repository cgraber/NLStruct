from distutils.core import setup, Extension
import numpy as np

module1 = Extension('fastmp', sources = ['fastmpmodule.cpp'],
                    include_dirs = [np.get_include()],
                    extra_compile_args = ['-Wall', '-W', '-fopenmp', '-O3', '-pedantic', '-std=c++0x', '-DWHICH_FUNC=1'])

setup(name = 'fastmp',
      version = '1.0',
      description = 'Runs message passing efficiently',
      ext_modules = [module1])
