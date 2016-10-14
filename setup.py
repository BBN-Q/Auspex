from setuptools import setup

setup(
    name='auspex',
    version='0.1',
    author='auspex Developers',
    package_dir={'':'src'},
    packages=[
        'auspex', 'auspex.instruments', 'auspex.filters'
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
        "networkx >= 1.11"
    ]
)
