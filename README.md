# pyEW
### Tools for finding equivalent widths of spectral lines


### Requires the following Python (2.7) libraries:
- numpy
- scipy
- matplotib
- PyQt5 (pyEW needs qt5Agg backend)



### Installation:

Copy or clone the github repository
to a directory of your choice and install with `python setup.py install`.

### Quick start:
`pyew your_config_file`

### Input files

- Stellar spectrum - an ascii file with wavelength 
in the first column and normalized flux in the second.
- Line list file - an ascii file with information on spectal lines. Each line of this file should include: wavelength, 
element (in MOOG format), excitation potential, log gf.
- List of files for analysis, full path to files with stellar spectrum, one per line.

### Config file
Config file includes all parameters required by pyEW.  Do no remove any keyword or section.
Copy this file to your project and edit it so it suits your needs.


`Input files` section:
- files_list - full path to file that includes a list of spectra for analysis.
There might be more than one spectrum listed on this list. 
- line_list_file - full path to a file that includes a list of lines for analysis.
 

`Spectrum section`:
- off - defines the range of spectrum around chosen line that will be analyzed. 
The range will be defined as: [x0-off, x0+off], where x0 is a center of line to be measured. 
- s_factor - a parameter for smoothing. Usually 3 or 4 is fine. T
his number should be lower if in defined range the number of points 
is low.
- rejt_auto - the value should be True or False. If True, signal to noise (SN) 
will be calculated based on ranges provided in the next setion of the config file.
The SN is required for local continuum fitting. If rejt_auto is set to False,
value specified by user will be used.
- local_cont - True or False - fit local continuum.
- rejt - parameter for local continuum fitting if rejt_auto = False. 
It is defined as 1 - 1/SN (see ARES paper for the definition od rejt:
 http://www.astro.up.pt/~sousasag/ares/2007_AA_Sousa.pdf )
- t_sig - threshold for hot pixels (t_sig * standard deviation of flux values) 

 `Lines ` section:
- r_lvl - minimum level for finding strong lines. This is applied to third derivative 
as r_lvl * standard deviation(3rd derivative)
and presented as two red,vertical lines on lower panel of the plot. 
The purpose of this is to control the number of lines introduced to multi gaussian fit.
- w_factor - area around center of line: x0 +/- fwhm * w_factor. Defines green-shaded region around 
line center 
- l_eqw - minimum value of equivalent width (EW) (in mA) to consider.
 This parameter is set to eliminate lines too week to beincluded in this analysis.
- h_eqw - maximum value of EW in mA. This parameter is  to eliminate lines too strong 
and possibly saturated that should not be included in the analysis.
- det_level - maximum difference (in A) between x0 and line position in spectrum.
- v_lvl - minimum reduced eqw that should be used for considering if Voigt profile
should be considered in analysis.If rEW>v_lvl, check Voigt profile.
- plot_flag - True or False - should I do plot for ALL lines?
- show_lines - show plot for some lines, 0 if no lines should be plotted, 
or  line1 line2 ... otherwise.



### Example

Go to `pyEW/example` catalog and open `sun.config` file for edition.
Then edit  `pyEW/example/spectra.list` so it includes FULL path to sun.asc file.
Run example files with:

`pyew sun.config`



### Output file
- is in format required by MOOG
- some extra information are placed at the end of line,
  like estimated error, width of a line (gaussian sigma),line depth.
  They will be ignored by MOOG, but maybe useful in identification
  of poorly modelled lines.
  
