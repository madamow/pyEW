import numpy as np
import scipy.optimize as so
from scipy.special import wofz

######################################################
# Gaussian fitting
######################################################


agauss = -4.0 * np.log(2.0)
c1 = 1.0692
c2 = 0.86639


def gaus(x_in, a, x0, fwhm):
    return a * np.exp(agauss * ((x_in - x0) / fwhm) ** 2)


def multiple_gaus(x, params):
    mg = np.zeros_like(x)
    for row in params:
        x0, a, fwhm = row
        mg = mg + gaus(x, a, x0, fwhm)
    return mg


def res_g(p, data):
    x, y = data
    a, x0, fwhm = p
    sg = gaus(x, a, x0, fwhm)
    err = 1. / np.abs(y)
    return (y - sg) / err


def res_mg(p, x, y, nb):
    params = np.reshape(p, (nb, 3))
    mg = multiple_gaus(x, params)
    err = 1. / np.abs(y)
    return (y - mg) / err


def get_gew(ag, fwhmg):
    # calculate EW for gaussian profile
    gew = 500. * ag * np.sqrt(np.pi / np.log(2)) * fwhmg
    return gew


def fit_single_gauss(x, f, a1, x01, fwhm1):
    gaus_p = so.leastsq(res_g, np.array([a1, x01, fwhm1]),
                        args=([x, 1.0 - f]), full_output=1)
    a1s, x01s, fwhm1s = np.abs(gaus_p[0])

    eqw_gf = get_gew(a1s, fwhm1s)

    gf_errs = leastsq_errors(gaus_p, 3)

    eqw_gf_err = eqw_gf * (gf_errs[0, 0] / a1s + gf_errs[0, 2] / fwhm1s)
    return eqw_gf, eqw_gf_err, gaus_p[0]


def fit_multi_gauss(x, f, strong_lines, det_level):
    params = np.ones((len(strong_lines), 3))
    params[:, 0] = strong_lines  # first columns = x0
    params[:, 2] = 0.05  # starting value for gauss fit

    new_params = np.array([])

    # First run
    while params.shape[0] != new_params.shape[0] and params.shape[0] > 0.:
        plsq = so.leastsq(res_mg, params,
                          args=(x, 1.0 - f, params.shape[0]), full_output=1)

        new_params = np.reshape(plsq[0], (params.shape[0], 3))

        # evaluate  run
        ind = np.where(np.abs(strong_lines - new_params[:, 0]) < det_level)

        strong_lines = np.array(strong_lines)[ind]
        params = new_params[ind]

    mg_errs = leastsq_errors(plsq, 3)  # fit quality

    return np.abs(params), mg_errs


def leastsq_errors(fit_tab, p_no):
    # Errors for parameters of the fit
    # so.leastsq result+no of parameters fitted
    # use so.leastsq output to estimate error parameter
    pcov = fit_tab[1]

    if pcov is None:
        row_col = int((len(fit_tab[0]) / p_no) * p_no)
        print("Covariance matrix is empty")
        errs_matrix = np.ones((row_col, row_col))
        errs_matrix[:, :] = 1000.0  # make errors huge
    else:
        sq = np.sum(fit_tab[2]['fvec'] ** 2) / (len(fit_tab[2]) - len(fit_tab[0]))
        errs_matrix = sq * pcov

    i = np.arange(len(fit_tab[0]))
    f_errs = np.reshape(errs_matrix[i, i] ** 2, (int(len(fit_tab[0]) / p_no), p_no))
    return f_errs


######################################################
# Voigt fitting
######################################################
def voigt(x, y):
    # The Voigt function is also the real part of
    # w(z) = exp(-z^2) erfc(iz), the complex probability function,
    # which is also known as the Faddeeva function. Scipy has
    # implemented this function under the name wofz()
    z = x + 1j * y
    I = wofz(z).real
    return I


def Voigt(nu, alphaD, alphaL, nu_0, A, a, b):
    # The Voigt line shape in terms of its physical parameters
    # alphaD, alphaL half widths at half max for Doppler and Lorentz(not FWHM)
    # A - scaling factor
    f = np.sqrt(np.log(2))
    x = (nu - nu_0) / alphaD * f
    y = alphaL / alphaD * f
    backg = a + b * nu
    V = A * f / (alphaD * np.sqrt(np.pi)) * voigt(x, y) + backg
    return V


def funcV(p, x):
    # Compose the Voigt line-shape
    a = 0.
    b = 0.
    alphaD, alphaL, nu_0, I = p
    return Voigt(x, alphaD, alphaL, nu_0, I, a, b)


def res_v(p, data):
    # Return weighted residuals of Voigt
    x, y, err = data
    err = 1. / np.abs(y)
    return (y - funcV(p, x)) / err


def fit_Voigt(x, f, nu_0):
    alpha_d = 0.01
    alpha_l = 0.01
    pv0 = [alpha_d, alpha_l, nu_0, 0.1]

    voigt_p = so.leastsq(res_v, pv0,
                         args=([x, 1.0 - f, np.ones_like(x)]), full_output=1)

    I = np.abs(voigt_p[0][3]) * 1000.
    v_errs = leastsq_errors(voigt_p, 4)[0][3]

    return I, v_errs, voigt_p[0]


def voigt_fwhm(alpha_d, alpha_l):
    v_fwhm = c1 * alpha_l + np.sqrt(c2 * alpha_l**2 + 4. * alpha_d**2)
    return v_fwhm
