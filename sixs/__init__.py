"""
sixs — Python translation of the 6S radiative transfer model.

Second Simulation of the Satellite Signal in the Solar Spectrum,
Version 4.1.  Original Fortran code by Vermote, Tanré, Déuzé,
Herman & Morcrette.

This translation includes the Estonian customisations (m_* variables)
from the original main.f.

Usage
-----
Command-line (stdin/stdout, same interface as original Fortran):

    python -m sixs < input.txt > output.txt

Programmatic:

    from sixs.sixs_main import run
    import io

    input_text = \"\"\"
    0
    30.0 0.0 0.0 0.0 7 1
    2
    0.0 0.344
    1
    0.70 0.29 0.00 0.01
    23.0
    0.0
    -1000
    6
    0
    0
    0
    0.2
    -2.0
    \"\"\"

    results = run(io.StringIO(input_text))
    print(results['apparent_reflectance'])
"""

from .sixs_main import run, main
from .commons   import atm, delta_sigma, aer, disc, ffu

__all__ = ['run', 'main', 'atm', 'delta_sigma', 'aer', 'disc', 'ffu']
__version__ = '4.1.0'
