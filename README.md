# pyEW
Python functions for finding equivalent widths of spectral lines

# What do you need?
 To use pyEW you need Python 2.7 and those libraries:

*numpy
*scipy (version 0.13 or later)
*matplotlib
*PyQt5 (pyEW needs qt5Agg backend)

Install instructions can be found here:
http://scipy.org/install.html
http://matplotlib.org/users/installing.html

If you are a python user, you probably have those packages already installed. If python is new to you, I recomend anaconda or miniconda for installation and updates.
https://www.continuum.io/downloads
http://conda.pydata.org/miniconda.html


Stellar spectrum
 - must be in ascii file,
 - must be in laboratory wavelength scale
 - it should be normalized (the continuum fit does not have to be perfect,
    continuum will be corrected locally)

Line list
- pyEW is written to give an output file ready for MOOG,
  hence in line list you need to include:
  wavelength, element (MOOG format), excitation potential, log gf

Output file
- is in format required by MOOG
- some extra information are placed at the end of line,
  like estimated error, width of a line (gaussian sigma),line depth.
  They will be ignored by MOOG, but maybe useful in identification
  of poorly modelled lines.
  
