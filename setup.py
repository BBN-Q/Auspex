import sys
import os
from setuptools import setup

# See https://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
if sys.version_info < (3,6):
    sys.exit("Sorry, Python < 3.6 is not supported by Auspex.")

install_requires = [
    "numpy >= 1.11.1",
    "scipy >= 0.17.1",
    "PyVISA >= 1.8",
    "tqdm >= 4.7.0",
    "pandas >= 0.18.1",
    "networkx >= 1.11",
    "matplotlib >= 2.0.0",
    "psutil >= 5.0.0",
    "cffi >= 1.11.5",
    "scikit-learn >= 0.19.1",
    "pyzmq >= 16.0.0",
    "pyusb >= 1.0.2",
    "pydotplus >= 2.0.0",
    "python-usbtmc >= 0.8"
]

#Use PyVISA-Py if running on Linux or MacOS
if os.name == "posix":
    install_requires.append("PyVISA-Py >= 0.2")
    install_requires.append("pyserial >= 3.4")

setup(
    name='auspex',
    version='0.4',
    author='auspex Developers',
    package_dir={'':'src'},
    packages=[
        'auspex', 'auspex.instruments', 'auspex.filters', 'auspex.analysis', 'auspex.qubit'
    ],
    scripts=[],
    description='Automated system for python-based experiments.',
    long_description=open('README.md').read(),
    install_requires=install_requires
)
