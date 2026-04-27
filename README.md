This is a python translation of the original scalar sixs fortran code. The original code came without a copyright notice or a readme file. Likely, it is not copyrighted - and neither is the python translation. I have no recollection where I downloaded it from decades ago. It has almost no dependencies, so use it as you like. There are no promises or guarantees. The translation was done in early 2026 by Claude (Sonnet 4.6) with minimum human help or oversight. Claude checked and debugged it itself, see 6S\_Python\_Conversion\_Notes.md for details. The code is slow and unoptimized, for any production environment the fortran version (an updated version of it!) should be used which will provide better accuracy and efficiency. The purpose of this translation is enabling quick ad-hoc test runs without learning (the Windows installation of) fortran 77, e.g. in educational use. While not the latest version, the results should still be mostly valid. I used  and translated the code without any intent to improve it. The python translation is made available for reference only (e.g., of Claude's capability in 2026) without any claims of usability. I am not an expert in atmospheric radiative transfer.



To install, a simple "pip install -e ." in the folder containing this file should do.



The original reference to whom all credit is due should be

Vermote, E.F., Tanré, D., Deuzé, J.L., Herman, M., \& Morcrette, J.-J. (1997), Second Simulation of the Satellite Signal in the Solar Spectrum, 6S: An Overview, IEEE Transactions on Geoscience and Remote Sensing, Vol. 35, No. 3, p. 675-686.

There are later developments (e.g., 6SV) which are not included here. I used the code to get spectral irradiances (among other things), so they are now output by the fortran and python versions. The small edits I made are marked in the code with comments.



Despite a lack of active development, bugfixes are still welcome. 



facilitator of the translation,

Matti Mõttus
matti.mottus@gmail.com

PS - I hope it's clear there is no need to acknowledge me or Claude here, all credits go to the original authors.

