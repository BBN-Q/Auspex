from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("aps3_wavepack", ["aps3_wavepack.pyx"], include_dirs=[numpy.get_include()])]

setup(
    name = "aps3_wavepacker",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)
