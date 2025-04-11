
import config

import numpy as np
from scipy.integrate import quad
from scipy.special import jv
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.stats import gaussian_kde
import emcee
from os.path import isfile
import matplotlib.pyplot as plt
import warnings



x_range, y_range = (-12, 6, .01), (-12, 14, .01)

def gauss_kde_pdf(posterior):
    # Return a function, computing the pdf with the given posterior.
    return gaussian_kde(posterior).pdf

def _KT18_integ(alpha, beta):
    return (2. / np.log(1 + beta / alpha) * quad(lambda x:
            np.exp(-alpha * x**2) * (1 - np.exp(-beta * x**2)) *
            jv(0, x) / x, 0, np.inf)[0])

def tab(name='KT18_table.npy', x_range=x_range, y_range=y_range):
    a, b = np.arange(*x_range), np.arange(*y_range)
    x, y = np.meshgrid(a, b)
    c = np.zeros(len(a) * len(b))
    n = 0
    for ln_alpha in a:
        for ln_beta in b:
            c[n] = max(_KT18_integ(np.e ** ln_alpha, np.e ** ln_beta), 1e-10)
            n += 1
            if n % 10000 == 0: print(n)
    mat = np.stack([x.T.reshape(-1), y.T.reshape(-1), c], axis=1)
    np.save(name, mat)

if isfile('KT18_table.npy'):
    _, _, z = np.load('KT18_table.npy').T
    x, y = np.arange(*x_range), np.arange(*y_range)
    rbs_func = RBS(x, y, z.reshape([len(x), len(y)]))
else:
    raise ValueError("Please load 'KT18_table.npy'!")
    # tab()


def KT18_model(x_array, beam, x0, KappaTstar):
    alpha = (beam ** 2 / 2 + x0 ** 2) / x_array ** 2
    beta = 2 * KappaTstar / x_array ** 2
    res = []
    for a, b in zip(alpha, beta):
        res.append(np.diagonal(rbs_func(np.log(a), np.log(b)))[0])
    return np.array(res)
    # ln_alpha, ln_beta = np.log(alpha[::-1]), np.log(beta[::-1])
    # return np.diagonal(rbs_func(ln_alpha, ln_beta))[::-1]


def _ln_likelihood(theta, x, y, yerr):
    sigma, x0, KappaTstar, f = theta
    model = KT18_model(x, sigma, x0, KappaTstar)
    ln_prob = -.5 * np.sum((y - model / f) ** 2 / yerr ** 2 + 2 * np.log(yerr))
    return ln_prob

def _ln_prior(theta, beam):
    sigma, x0, KappaTstar, f = theta
    sigma_0, sigma_u = beam
    if 0 < x0 * 1e3 < 200 and 0 < KappaTstar < 1 and 1 < f < 5 and sigma > 0:
        return -.5 * ((sigma - sigma_0) ** 2 / sigma_u ** 2 +
                      np.log(sigma_u ** 2 * f ** 2))
    return -np.inf


def _ln_prob(theta, x, y, yerr, beam):
    p = _ln_prior(theta, beam)
    l = _ln_likelihood(theta, x, y, yerr)
    if not np.isfinite(p) or np.isnan(p) or not np.isfinite(l) or np.isnan(l):
        return -np.inf
    return p + l


def fit(x, y, yerr, beam, kt_prior=None, n_dim=4,
        n_walker=config.n_walker, n_step=config.n_step, n_sample=config.n_sample):
    
    pos = (np.array([beam, 0.08, 0.08, 2.5]) + np.random.randn(n_walker, n_dim) *
           np.array([0., 0.01, 0.01, 0.6]))
    sampler = emcee.EnsembleSampler(n_walker, n_dim, _ln_prob,
                                    args=(x[1:], y[1:], yerr[1:], (beam, 1e-6)))
    # The first term of y should always be zero and
    # will never be affected by the factor of f.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        sampler.run_mcmc(pos, n_step, progress=True)
    samples = sampler.get_chain()
    flat_samples = samples[-n_sample:, :, :]
    perc_samples = flat_samples.reshape((-1, n_dim))
    par50 = np.percentile(perc_samples, 50, axis=0)
    par16 = np.percentile(perc_samples, 16, axis=0)
    par84 = np.percentile(perc_samples, 84, axis=0)

    print(par50)
    print(par16)
    print(par84)

    return perc_samples, par50





