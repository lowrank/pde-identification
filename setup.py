from distutils.core import setup, Extension

import numpy

module = Extension("design_matrix", ["utilities/design_matrix.pyx"],
                   include_dirs=[numpy.get_include()])
setup(
    name='cythonTest',
    version='1.0',
    author='jetbrains',
    ext_modules=[module],
    include_dirs=[numpy.get_include()]
)
