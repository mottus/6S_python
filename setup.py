from setuptools import setup, find_packages

setup(
    name='sixs',
    version='4.1.0',
    description='6S Radiative Transfer Model — Python translation from Fortran 77',
    long_description=(
        'Second Simulation of the Satellite Signal in the Solar Spectrum (6S), '
        'version 4.1.  Translated from Fortran 77 to Python.  '
        'Includes Estonian customisations from the original main.f.'
    ),
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['numpy'],
    entry_points={
        'console_scripts': ['sixs=sixs.sixs_main:main'],
    },
)
