
import constant

import numpy as np


# ========== utils for derendening ==========

def _kappa(wavelength):
    Rv = 4.05  # Rv will be cancelled once _kappa(a) - _kappa(b)
    x = 1e4 / wavelength
    if wavelength >= 6300:
        return 2.659 * (-1.857 + 1.040*x) + Rv
    else:
        return 2.659 * (-2.156 + 1.509*x - 0.198*x**2 + 0.011*x**3) + Rv

def dered_f(EBV, wavelength, reddest_wavelength):
    kappa_a = _kappa(wavelength)
    kappa_b = _kappa(reddest_wavelength)
    factor = 10 ** (0.4 * (kappa_a - kappa_b) * EBV)
    return factor


# From Perez-Montero (2017)

def _t2abd(t, a, b, c, d=0):
    ne = 1e2
    return a + b / t + c * np.log10(t) + d * ne

def oxygen(galaxy):
    OII = 10 ** (np.log10(galaxy.ratio(['OII7320', 'OII7330'], ['Hbeta'])) +
                 _t2abd(galaxy.t_OII_i, 7.21, 2.511, -0.422, d=0.000398) - 12)
    OIII = 10 ** (np.log10(galaxy.ratio(['OIII5007'], ['Hbeta']) * 4. / 3.) +
                  _t2abd(galaxy.t_OIII_i, 6.1868, 1.2491, -0.5816) - 12)
    Z = 12 + np.log10(OII + OIII)
    return Z, ~np.isnan(Z)

def oxygen_small(galaxy):
    OII = 10 ** (np.log10(galaxy.ratio(['OII7320', 'OII7330'], ['Hbeta'])) +
                 _t2abd(galaxy.t_OII_i, 7.21, 2.511, -0.422, d=0.000398) - 12)
    OIII = 10 ** (np.log10(galaxy.ratio(['OIII5007'], ['Hbeta']) * 4. / 3.) +
                  _t2abd(galaxy.t_OIII_i, 6.1868, 1.2491, -0.5816) - 12)
    Z = 12 + np.log10(OII + OIII)
    return Z, ~np.isnan(Z)


def nitrogen(galaxy):
    NII = 10 ** (np.log10(galaxy.ratio(['NII6584'], ['Hbeta']) * 4. / 3.) +
                 _t2abd(galaxy.t_NII_i, 6.291, 0.90221, -0.5511) - 12)
    Z = 12 + np.log10(NII)
    return Z, ~np.isnan(Z)


def nitrogen_direct(galaxy):
    NII = 10 ** (np.log10(galaxy.ratio(['NII6584'], ['Hbeta']) * 4. / 3.) +
                 _t2abd(galaxy.t_NII, 6.291, 0.90221, -0.5511) - 12)
    Z = 12 + np.log10(NII)
    return Z, ~np.isnan(Z)


def sulphur(galaxy):
    SII = 10 ** (np.log10(galaxy.ratio(['SII6717', 'SII6731'], ['Hbeta'])) +
                 _t2abd(galaxy.t_OII_i, 5.463, 0.941, -0.37) - 12)
    SIII = 10 ** (np.log10(galaxy.ratio(['SIII6312'], ['Hbeta'])) +
                  _t2abd(galaxy.t_SIII, 6.695, 1.664, -0.513) - 12)
    Z = 12 + np.log10(SII + SIII)
    return Z, ~np.isnan(Z)

def sulphur_small(galaxy):
    SII = 10 ** (np.log10(galaxy.ratio(['SII6717', 'SII6731'], ['Hbeta'])) +
                 _t2abd(galaxy.t_OII_i, 5.463, 0.941, -0.37) - 12)
    SIII = 10 ** (np.log10(galaxy.ratio(['SIII6312'], ['Hbeta'])) +
                  _t2abd(galaxy.t_SIII, 6.695, 1.664, -0.513) - 12)
    Z = 12 + np.log10(SII + SIII)
    return Z, ~np.isnan(Z) & nitrogen_direct(galaxy)[1]




