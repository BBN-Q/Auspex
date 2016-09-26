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
        "Numpy >= 1.6.1",
        "pandas >= 0.14",
    ]
)
