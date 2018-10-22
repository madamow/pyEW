import ConfigParser
import numpy as np
from Gauss_Voigt import get_gew


# INPUT
def read_config_file(cfile):
    # Load config file
    config = ConfigParser.ConfigParser()
    config.read(cfile)

    return config


def get_lab_data(cfile):
    # Load a file with data for lines to analyze
    # Format: line element excitation_potential loggf,
    # all in MOOG like format
    return np.loadtxt(cfile.get('Input files', 'line_list_file'), usecols=[0, 1, 2, 3])


def get_flist(cfile):
    flist = np.genfromtxt(cfile.get('Input files', 'files_list'), dtype=None)
    # Add a end line
    # This is for cases with only one spectra on list. It is just a trick
    # to make iteration possible
    if flist.size == 1:
        flist = np.append(flist, ["end_iter"])
    return flist


#######################################################################################################################
# PRINT ON SCREEN
def moog_entry(l, ew, eew):
    moog = "%10s%10s%10s%10s%10s%10s%10.2f %6.3e \n" % \
          (l[0], l[1], l[2], l[3], '', '', ew, eew)
    return moog


def print_mgauss_data(rslt, logfile):
    # print full data for all lines
    # fitted with multi Gauss function
    mg_params = rslt['mg'][1]
    print_and_log(logfile, ["\n", mg_params.shape[0],
                            "lines in multi gaussian fit:"])
    for gfit in mg_params:
        ew = get_gew(gfit[1], gfit[2])
        # Info about lines in multigaussian fit
        info = "%4.2f %s%4.2f %s%4.3f %s%4.2f %s%4.2f" % \
               (gfit[0], "depth=", gfit[1],
                "FWHM=", gfit[2],
                "EW=", ew,
                "RW=", np.log10(0.001 * ew / gfit[0]))
        print_and_log(logfile, [info])


def print_line_info(rslt, logfile):
    fit_labels = {'mg': 'multi Gauss', 'sg': 'part of mGauss', 'g': 'Gauss', 'v': 'Voigt'}
    for fit in rslt:
        f_info = "%15s %s %4.2f %s %f %s %f" % \
                (fit_labels[fit], ": EW =", rslt[fit][2],
                 "fq =", rslt[fit][3], "o-c:", rslt[fit][4])
        print_and_log(logfile, [f_info])


#######################################################################################################################
# OUTPUT

# OUTPUT : LOGFILE
def create_log(fname):
    logfile = open(fname.split("/")[-1].split(".")[0] + ".log", 'w')
    return logfile


def print_and_log(logfile, list_of_inps):
    s = ' '.join(map(str, list_of_inps))
    print s
    logfile.write(s + "\n")


# OUTPUT : OUT FILE
def create_out(fname, cfile):
    # Create an output file
    out_name = "moog_" + \
               cfile.get('Input files', 'line_list_file').split("/")[-1] + "_" + \
               fname.split("/")[-1].split(".")[0] + ".out"

    return open(out_name, 'w')


def moog_output(out_file, logfile, a_line, lr):
    # Write to output file
    moog = moog_entry(a_line, lr[3], lr[4])
    out_file.write(moog)
    print_and_log(logfile, ["\nMoog entry:\n", moog])
