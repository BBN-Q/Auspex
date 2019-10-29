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
    "pandas >= 0.18.1",
    "networkx >= 1.11",
    "matplotlib >= 2.0.0",
    "psutil >= 5.0.0",
    "cffi >= 1.11.5",
    "scikit-learn >= 0.19.1",
    "pyzmq >= 16.0.0",
    "pyusb >= 1.0.2",
    "python-usbtmc >= 0.8",
    "ipykernel>=5.0.0",
    "ipywidgets>=7.0.0",
    "sqlalchemy >= 1.2.15",
    "setproctitle",
    "serial",
    "progress"
]

#Use PyVISA-Py if running on Linux or MacOS
if os.name == "posix":
    install_requires.append("PyVISA-Py >= 0.3")
    install_requires.append("pyserial >= 3.4")

# python setup.py sdist
# python setup.py bdist_wheel
# For testing:
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# For distribution:
# twine upload dist/*
# Test with:
# pip install --extra-index-url https://test.pypi.org/simple/ auspex

setup(
    name='auspex',
    version='2019.2',
    author='Auspex Developers',
    package_dir={'':'src'},
    packages=[
        'auspex', 'auspex.instruments', 'auspex.filters', 'auspex.analysis', 'auspex.qubit'
    ],
    scripts=[],
    url='https://github.com/BBN-Q/auspex',
    download_url='https://github.com/BBN-Q/auspex',
    license="Apache 2.0 License",
    description='Scientific measurement platform specifically geared towards superconducting qubit measurements.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6',
    keywords="quantum qubit pipeline measure instrument experiment control automate plot"
)
