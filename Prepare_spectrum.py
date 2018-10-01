import numpy as np
from IO_functions import print_and_log
from scipy import ndimage
from scipy.signal import argrelextrema

# Prepare observed spectrum for analysis


def do_linear(spec):
    # Transforms spectrum to linear scale
    first = spec[0, 0]
    last = spec[-1, 0]
    step = (last - first) / spec.shape[0]  # step in wavelength
    x = np.arange(first, last, step)
    y = np.interp(x, spec[:, 0], spec[:, 1])
    lin_spec = np.transpose(np.vstack((x, y)))
    return lin_spec, step


# Get Signal to Noise Ratio from spectrum or config file
# Calculate from spectrum
def read_cont_sec(cfile):
    secs = []
    for elem in cfile.items('cont_sec'):
        secs.append(elem[1].split(' '))
    return secs


def calculate_snr(spec, cfile, logfile):
    print_and_log(logfile, ['rejt parameter was found automatically'])

    rejt_reg = np.array(read_cont_sec(cfile), dtype=float)
    rejt_tab = []

    for item in rejt_reg:
        reg, rres = do_linear(spec[np.where((spec[:, 0] > item[0]) & (spec[:, 0] < item[1]))])
        reg_smo = ndimage.gaussian_filter1d(reg[:, 1],
                                            sigma=cfile.getfloat('Spectrum', 's_factor'),
                                            mode='wrap')
        rejt_tab.append(np.std(reg[:, 1] - reg_smo))

    rejt = 1. - np.median(rejt_tab)
    snr = 1. / (1. - rejt)

    return rejt, snr

def get_snr(spec, cfile, logfile):
    # Check if rejt parameter is defined by user.
    # If not - calculate it from observed spectrum

    if not cfile.getboolean('Spectrum', 'rejt_auto'):
        rejt, snr = calculate_snr(spec, cfile, logfile)

    else:
        print_and_log(logfile, ['rejt parameter defined by user'])
        rejt = cfile.getfloat('Spectrum', 'rejt')
        snr = 1.0 / (1.0 - rejt)

    print_and_log(logfile, ['rejt parameter:', '%4.3f' % rejt])
    print_and_log(logfile, ['signal to noise:', '%4.0f' % snr])

    return rejt, snr


# Remove hot pixels
def hot_pixels(spec, t_sigma, logfile):
    # Cut hot pixels from the top
    thresh = t_sigma * np.std(spec[:, 1]) + 1.0
    print_and_log(logfile, ['Points with flux values >', '%4.2f' % thresh,
                            'will be removed from continuum fit'])

    hot_no = spec[np.where(spec[:, 1] > thresh)].shape[0]
    print_and_log(logfile, [hot_no, 'point(s) removed'])

    clean = spec[np.where(spec[:, 1] < thresh)]

    return clean

def smooth(y, box_pts):
    #smooths function, recipe from StackOverflow
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def local_continuum(spec, rejt, t_sigma, logfile):
    # Find local continuum level

    spec = hot_pixels(spec, t_sigma, logfile)

    stop = False
    tab = np.copy(spec)
    p_no = 0
    iter = 0.0
    while not stop:
        iter = iter + 1.
        p_no_prev = p_no
        ft = np.polyfit(tab[:, 0], tab[:, 1], 2)
        tab = spec[np.where(spec[:, 1] > rejt * np.polyval(ft, spec[:, 0]))]
        p_no = tab.shape[0]

        if p_no == p_no_prev or iter > 100.:
            stop = True
    if p_no != 0 and iter < 100.:
        cont = spec[:, 1] / np.polyval(ft, spec[:, 0])
        spec[:, 1] = cont
    else:
        print_and_log(logfile, ['Fail when correcting continuum'])
        # This is for cases where line is located
        # i.e. in a break of echelle spectrum
    return spec


def derivatives(flux, dx, s_factor):
    dxdxdx = dx * dx * dx
    # First derivative
    gf = ndimage.gaussian_filter1d(flux, sigma=s_factor, order=1, mode='wrap') / dx

    # Second derivative
    ggf = ndimage.gaussian_filter1d(flux, sigma=s_factor, order=2, mode='wrap') / (dx * dx)

    # Third derivative
    gggf = np.array(ndimage.gaussian_filter1d(flux, sigma=s_factor, order=3, mode='wrap') / dxdxdx)

    return gf, ggf, gggf

def get_thold(gggf, r_lvl):
    return np.std(gggf) * r_lvl

def find_inflection(x, y):
    # This function finds zero points for set of data
    # It uses linear interpolation between two points with
    # different signs. Then it selects points which change their sign
    #  from + to -, and form - to +
    to_minus = []
    to_plus = []

    for i in np.arange(0, len(y) - 1, 1):
        a = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        b = y[i] - a * x[i]
        if y[i] > 0.0 and y[i + 1] < 0.0:
            to_minus.append(-b / a)
        elif y[i] < 0.0 and y[i + 1] > 0:
            to_plus.append(-b / a)

    to_minus = np.array(to_minus)
    to_plus = np.array(to_plus)

    return to_minus, to_plus

def find_flex_points(spec, dx, cfile):
    # Find derivatives
    gf, ggf, gggf = derivatives(spec[:, 1], dx, cfile.getfloat('Spectrum', 's_factor'))

    # Find inflection points
    gggf_infm, gggf_infp = find_inflection(spec[:, 0], gggf)

    return gggf_infm, gggf_infp, gggf

def evaluate_lines(line, strong_lines, det_level, gggf_infm, logfile):
    # check if our line was identified
    if len(strong_lines) == 0 and (np.abs(gggf_infm - line).min() > det_level):
        print_and_log(logfile, [line, 'was not detected'])
    elif len(strong_lines) == 0 and (np.abs(gggf_infm - line).min() < det_level):
        # no strong lines detected
        print_and_log(logfile, ['I see no strong lines here,but a weak line close to', line, 'was detected'])
        strong_lines.append(line)
    elif (len(strong_lines) > 0 and np.abs(gggf_infm - line).min() < det_level and
          np.abs(strong_lines - line).min() < det_level):
        print_and_log(logfile, ['line', line, ' was detected and it was classified as a strong line'])
        pass
    elif (len(strong_lines) > 0 and (np.abs(gggf_infm - line).min() < det_level) and
          np.abs(strong_lines - line).min() > det_level):
        print_and_log(logfile, ['I see this line at', line, ', but it is weak'])
        strong_lines.append(line)
    else:
        print_and_log(logfile, [line, 'was not detected'])
    return strong_lines


def find_strong_lines(x, xo, gggf, gggf_infm, cfile, logfile):
    # We need to find strong lines in spectrum and ingnore all small changes in flux.
    # Here - code checks the change in signal around inflection points and compares it
    # to noise multiplied by rejection parameter

    max_ind = argrelextrema(gggf, np.greater)
    min_ind = argrelextrema(gggf, np.less)

    max_tab = np.array([x[max_ind], gggf[max_ind]])
    min_tab = np.array([x[min_ind], gggf[min_ind]])

    thold = get_thold(gggf, cfile.getfloat('Lines', 'r_lvl'))
    str_lines = []

    if not (max_tab.size != 0 and min_tab.size != 0 and thold == 0.0):
        for item in gggf_infm:
            indx = np.abs(max_tab[0, :] - item).argmin()
            indm = np.abs(min_tab[0, :] - item).argmin()

            if ((np.abs(max_tab[1, indx]) > thold) and
                    (np.abs(min_tab[1, indm]) > thold)):
                str_lines.append(item)

    str_lines = evaluate_lines(xo, str_lines,
                               cfile.getfloat('Lines', 'det_level'),
                               gggf_infm, logfile)
    return str_lines





