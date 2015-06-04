# pyEW
Python functions for finding equivalent widths of spectral lines

To run this program you need those python libraries:

*numpy
http://docs.scipy.org/doc/numpy/user/install.html

*scipy 
http://www.scipy.org/scipylib/download.html

*matplotlib
http://matplotlib.org/users/installing.html


#Stellar spectrum
 - must be in ascii file,
 - must be in laboratory wavelength scale

#Line list
- pyEW is written to give an output file ready for MOOG,
  hence in line list you need to include:
  wavelength, element (MOOG format), excitation potential, log gf

#Output file
- is in format reqired by MOOG
- some extra information are placed at the end of line,
  like estimated error, width of a line (gaussian sigma),line depth.
  They will be ignored by MOOG, but maybe useful in identification
  of poorly modelled lines.
  