
from setuptools import setup

setup(
    name='pycontrol',
    version='0.1',
    author='pycontrol Developers',
    package_dir={'':'src'},
    packages=[
        'pycontrol', 'pycontrol.instruments'
    ],
    scripts=[],
    description='Control things with Python',
    long_description=open('README.md').read(),
    install_requires=[
        "Numpy >= 1.6.1",
        "pandas >= 0.14",
    ]
)