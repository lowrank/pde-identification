from distutils.core import setup, Extension

import numpy
from Cython.Build import cythonize

module = Extension("design_matrix", ["utilities/design_matrix.pyx"],
                   include_dirs=[numpy.get_include()])
setup(
    ext_modules=cythonize("utilities/design_matrix.pyx", include_path=[numpy.get_include()]),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'numpy>=1.10',
        'scipy>=0.13',
        'Cython>=0.23',
        'sympy',
        'matplotlib'],
)
