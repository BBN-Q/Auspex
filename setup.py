import sys
from setuptools import setup

# See https://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
if sys.version_info < (3,6):
    sys.exit("Sorry, Python < 3.6 is not supported by Auspex.")

setup(
    name='auspex',
    version='0.1',
    author='auspex Developers',
    package_dir={'':'src'},
    packages=[
        'auspex', 'auspex.instruments', 'auspex.filters', 'auspex.analysis'
    ],
    scripts=[],
    description='Automated system for python-based experiments.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.11.1",
        "scipy >= 0.17.1",
        "PyVISA >= 1.8",
        "h5py >= 2.6.0",
        "tqdm >= 4.7.0",
        "pandas >= 0.18.1",
        "networkx >= 1.11",
        "matplotlib >= 2.0.0",
        "ruamel.yaml >= 0.15.18"
    ]
)
