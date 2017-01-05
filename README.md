# pyEW
Python functions for finding equivalent widths of spectral lines

What do you need?
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

Prepare config file
Open eqw.config file for editing. Do not remove any keyword or section. 
In Input files section:
- files_list - a list of files with stellar spectrum. There might be more than one.
Stellar spectrum has to be written as an ascii file.
It should be normalized and in laboraroty wavelength scale

- line_list_file - pyEW is written to give an output file ready for MOOG,
  hence in line list you need to include: wavelength, element (MOOG format), excitation potential, log gf.
  

In Spectrum section:
- off - defines the range of spectrum around chosen lines that will be analyzed. The range will be defined as: [x0-off, x0+off], x0 is a center of line to be measured. 
- s_factor - smoothing. Usually 3 or 4 is fine. This number should be lower if in defined range the numer of points is low (low resolution spectra, or very narrow range).

- rejt_auto - the value should be True or False. If True, signal to noise (SN) will be calculated based on ranges provided in the next setion of the config file. The SN is required for local continuum fitting. If rejt_auto is set to False, value specified by user will be used.

- rejt - parameter for local continuum fitting if rejt_auto = False. It is defined as 1 - 1/SN (see also ARES paper)

- t_sig - threshold for hot pixels. 


Output file
- is in format required by MOOG
- some extra information are placed at the end of line,
  like estimated error, width of a line (gaussian sigma),line depth.
  They will be ignored by MOOG, but maybe useful in identification
  of poorly modelled lines.
  
