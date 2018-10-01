import numpy as np
from Gauss_Voigt import *
from IO_functions import *

def pm_width(x, x01, s1, w_factor):  # s1 is fwhm
    iu = np.abs(x - x01 - w_factor * np.abs(s1)).argmin()
    il = np.abs(x - x01 + w_factor * np.abs(s1)).argmin()

    if iu == il or np.abs(iu - il) < 3.:
        il = 0
        iu = len(x)
    return il, iu


def append_to_dict(tab, lbl, fc, param, eqw, eqw_err, soc):
    tab[lbl].append(fc)
    tab[lbl].append(param)
    tab[lbl].append(eqw)
    tab[lbl].append(eqw_err)
    tab[lbl].append(soc)
    return tab


def find_eqws(line, spec, strong_lines, cfile):
    results = {'mg': [], 'sg': [], 'g': [], 'v': []}
    # Fit multiple gaussian profile
    params, mg_errs = fit_multi_gauss(spec[:, 0], spec[:, 1], strong_lines,
                                      cfile.getfloat('Lines', 'det_level'))

    if params.shape[0] == 0:
        print 'Line ', line, 'was not detected'
        params = np.array([[-9.9, -9.9, -9.9]])

    mgaus = multiple_gaus(spec[:, 0], params)

    ip = np.abs(params[:, 0] - line).argmin()
    x01, a1, s1 = params[ip, :]

    eqw = get_gew(a1, s1)
    eqw_err = eqw * (mg_errs[ip, 0] / a1 + mg_errs[ip, 2] / s1)

    # Calculate single gauss profile
    # (this gauss is a part of multigaussian fit)
    sgaus = gaus(spec[:,0], a1, x01, s1)
    sparams = [a1, x01, s1]

    # Determine region close to gaussian line center#
    il, iu = pm_width(spec[:,0], x01, s1, cfile.getfloat('Lines', 'w_factor'))

    oc_mg = np.std(np.abs(spec[il:iu, 1] - 1.0 + mgaus[il:iu]))
    oc_sg = np.std(np.abs(spec[il:iu, 1] - 1.0 + sgaus[il:iu]))

    results = append_to_dict(results, 'mg', mgaus, params, eqw, eqw_err, oc_mg)
    results = append_to_dict(results, 'sg', sgaus, sparams, eqw, eqw_err, oc_sg)

    # Fit single Gauss and Voigt profile
    eqw_gf, eqw_gf_err, gparams = fit_single_gauss(spec[il:iu, 0], spec[il:iu, 1], a1, x01, s1)
    gausf = gaus(spec[:, 0], gparams[0], gparams[1], gparams[2])
    oc_g = np.std(np.abs(spec[il:iu, 1] - 1.0 + gausf[il:iu]))
    results = append_to_dict(results, 'g', gausf, gparams, eqw_gf, eqw_gf_err, oc_g)

    try:
        I, v_errs, vpar = fit_voigt(spec[il:iu, 0], spec[il:iu, 1], x01)
    except:
        I, v_errs, vpar = fit_Voigt(spec[:, 0], spec[:, 1], x01)
    svoigt = Voigt(spec[:, 0], vpar[0], vpar[1], vpar[2], vpar[3], 0., 0.)
    oc_v = np.std(np.abs(spec[il:iu, 1] - 1.0 + svoigt[il:iu]))
    results = append_to_dict(results, 'v', svoigt, vpar, I, v_errs, oc_v)

    return results


def evaluate_results(line, rslt, cfile, logfile):
    print_line_info(rslt, logfile)

    if rslt['mg'][3] > 0.5 * rslt['mg'][2]:
        print_and_log(logfile, ['Huge error!', rslt['mg'][2], rslt['mg'][3]])
        hu = True
    else:
        hu = False

    if rslt['v'][4] < rslt['mg'][4] and \
            rslt['v'][4] < rslt['g'][4] and \
            np.log10(rslt['mg'][2] * 0.001 / line) > cfile.getfloat('Lines', 'v_lvl'):
        print_and_log(logfile, ['using Voigt profile'])
        v_fwhm = voigt_fwhm(rslt['v'][1][1], rslt['v'][1][2])
        out = [line, rslt['v'][1][2], v_fwhm, rslt['v'][2], rslt['v'][3]]

    elif rslt['g'][4] < rslt['mg'][4] and \
            np.log10(rslt['mg'][2] * 0.001 / line) < cfile.getfloat('Lines', 'v_lvl'):
        print_and_log(logfile, ['using single Gauss fit'])
        out = [line, rslt['g'][1][1], rslt['g'][1][2], rslt['g'][2], rslt['g'][3]]

    elif hu and rslt['g'][4] < rslt['sg'][4]:
        print_and_log(logfile, ['using single Gauss fit'])
        out = [line, rslt['g'][1][1], rslt['g'][1][2], rslt['g'][2], rslt['g'][3]]
    else:
        out1 = rslt['mg'][1][np.abs(rslt['mg'][1][:, 0] - line).argmin()]
        out = [line, out1[0], np.abs(out1[2]), rslt['mg'][2], rslt['mg'][3]]

    if out[3] > cfile.getfloat('Lines', 'h_eqw') or out[3] < cfile.getfloat('Lines', 'l_eqw'):
        print_and_log(logfile, ['Line is too strong or too weak'])
        out[2] = -99.9
        out[3] = -99.9
        out[4] = 99.9

    if np.abs(line - out[1]) > cfile.getfloat('Lines', 'det_level'):
        print_and_log(logfile, [line, 'line outside the det_level range'])
        out[2] = -99.9
        out[3] = -99.9
        out[4] = 99.9

    return out


# EW analysis
def EW_analysis(line, spec, strong_lines, cfile, logfile):
    # do all fits: multi gauss, sgauss (part of multi gauss),
    # gauss fitted in small area, voigt fitted in small area
    r_tab = find_eqws(line, spec, strong_lines, cfile)  # results tab
    print_mgauss_data(r_tab, logfile)
    # Check if EW is reasonable
    lr = evaluate_results(line, r_tab, cfile, logfile)

    return r_tab, lr
