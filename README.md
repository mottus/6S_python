This is a python translation of the original scalar sixs fortran code. The original code came without a copyright notice or a readme file. Likely, it is not copyrighted - and neither is the python translation. I have no recollection where I downloaded it from decades ago. It has almost no dependencies, so use it as you like. There are no promises or guarantees. The translation was done in early 2026 by Claude (Sonnet 4.6) with minimum human help or oversight. Claude checked and debugged it itself, see 6S\_Python\_Conversion\_Notes.md for details. The code is slow and unoptimized, for any production environment the fortran version (an updated version of it!) should be used which will provide better accuracy and efficiency. 

Claude has fixed a large number of bugs introduced in the translation and its (mis)understanding of the code. A new solar irradiance spectrum is being used. I think it produces meaningful results, but may still not be correct. Claude has compared Fortran and Python 6S and reports that "The atmospheric physics (transmittances, spherical albedo, gas absorption) agree to better than 0.03%. The codes are equivalent for atmospheric correction purposes" -- but AI can hallucinate. It suggests that 3 harmonics is enough, too (speeds up computations). Note that the python version includes a newer solar irradiance spectrum. For hyperspectral image correction, numerical integration errors of approx 1-1.5 % (claiming fortran uses fixed-step trapezoidal) remain.

The purpose of this translation is enabling quick ad-hoc test runs without learning (the Windows installation of) fortran 77, e.g. in educational use. While not the latest version, the results should still be mostly valid. I used  and translated the code without any intent to improve it. The python translation is made available for reference only (e.g., of Claude's capability in 2026) without any claims of usability. I am not an expert in atmospheric radiative transfer.

Computational improvements were made by Claude (e.g., option to reduce the number of harmonics in SOS and parallel processing capacity). Further improvements may be introduced over time, but they may cause new bugs. For example, Claude estimates that a numba rewrite of os_sos will lead to ~30x improvement in computing time. This may happen at some point, but will introduce more dependencies.

The library contains scripts (functions) and Tk GUI tools for importing USGS Hyperion Geotiff zipfiles (converting to ENVI format) and correcting it atmospherically with user-provided AOD and water column values. Atmospheric properties can be retrieved from AERONET using a GUI tool. The tool can likely be used for other satelite images as well. Again, no guarantees on accuracy!

To install, a simple "pip install -e ." in the folder containing this file should do.

The original reference to whom all credit is due should be

Vermote, E.F., Tanré, D., Deuzé, J.L., Herman, M., \& Morcrette, J.-J. (1997), Second Simulation of the Satellite Signal in the Solar Spectrum, 6S: An Overview, IEEE Transactions on Geoscience and Remote Sensing, Vol. 35, No. 3, p. 675-686.

More references are in the code comments.

There are later developments (e.g., 6SV) which are not included here. I used the code to get spectral irradiances (among other things), so they are now output by the fortran and python versions. The small edits I made are marked in the code with comments.


Despite a lack of continuous active development, bugfixes are still welcome. 



facilitator of the translation,

Matti Mõttus
matti.mottus@gmail.com

PS - I hope it's clear there is no need to acknowledge me or Claude here, all credits go to the original authors.

