
from ctypes import *
import numpy as np
from scipy.stats import binned_statistic
import time
import os
import warnings
import platform

def _bin_array(r, f, bin_size=.1): # phase=0
    bin_edge = np.arange(min(r), max(r) + bin_size, bin_size)
    stat = binned_statistic(r, f, bins=bin_edge)
    mask = ~np.isnan(stat.statistic)
    bin_r = stat.bin_edges[:-1][mask]
    bin_f = stat.statistic[mask]
    return bin_r, bin_f

def _step(rad, met, bin_rad, bin_met):
    rad_matrix = abs(np.subtract.outer(rad, bin_rad))
    min_value = np.expand_dims(np.min(rad_matrix, axis=1), axis=1)
    # np.where will return ALL the min values!
    x, y = np.where(rad_matrix == min_value)
    # If having recurring items, remove them.
    y = y[np.insert(np.diff(x) != 0, 0, True)]
    step_func = bin_met[y]
    fluc = met - step_func
    return fluc

def _bin_stat(f, x, y, report=False, bin_size=.2, max_sep=5.):
    suffix = 'linux' if platform.platform().split('-')[0] == 'Linux' else 'macos'
    abs_path = os.path.dirname(os.path.abspath(__file__))
    lib = CDLL(abs_path + '/lib_' + suffix + '.so')
    three_arrays = np.concatenate([f, x, y], axis=0)
    lib.group_by.argtypes = (POINTER(c_float), c_int, c_int, c_int, c_float, c_float)
    lib.group_by.restype = POINTER(c_float)
    c_array = np.ctypeslib.as_ctypes(three_arrays.astype(np.float32))
    length = int(max_sep / bin_size) + 1
    t1 = time.time()
    c_res = lib.group_by(c_array, len(f), length, int(report), bin_size, max_sep)
    t2 = time.time()
    if report:
        print("\nTwo point correlation consumes %.2fs.\n" %(t2 - t1))
    py_res = cast(c_res, POINTER(c_float * (length * 2))).contents
    res = np.array(list(py_res), dtype=float).reshape(2, -1)
    return res[1]

def _cross_stat(f1, f2, x, y, report=False, bin_size=.2, max_sep=5.):
    suffix = 'linux' if platform.platform().split('-')[0] == 'Linux' else 'macos'
    abs_path = os.path.dirname(os.path.abspath(__file__))
    lib = CDLL(abs_path + '/lib_' + suffix + '.so')
    four_arrays = np.concatenate([f1, f2, x, y], axis=0)
    lib.cross_by.argtypes = (POINTER(c_float), c_int, c_int, c_float, c_float)
    lib.cross_by.restype = POINTER(c_float)
    c_array = np.ctypeslib.as_ctypes(four_arrays.astype(np.float32))
    length = int(max_sep / bin_size) + 1
    t1 = time.time()
    c_res = lib.cross_by(c_array, len(x), length, bin_size, max_sep)
    t2 = time.time()
    if report:
        print("\nCross correlation consumes %.2fs.\n" %(t2 - t1))
    py_res = cast(c_res, POINTER(c_float * (length * 2))).contents
    res = np.array(list(py_res), dtype=float).reshape(2, -1)
    return res[1]

def _tpcf(f, x, y, report=False, bin_size=.03, max_sep=5.):
    mean2, sigma2 = np.mean(f) ** 2, np.std(f) ** 2  # mean is 0
    bin_scorr = _bin_stat(f, x, y,
                          report=report, bin_size=bin_size, max_sep=max_sep)
    bin_d = np.arange(0, max_sep + bin_size, bin_size)[:len(bin_scorr)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        bin_s = (bin_scorr - mean2) / sigma2
    return bin_d, bin_s

def _ccf(f1, f2, x, y, report=False, bin_size=.03, max_sep=5.):
    mean1, mean2, sigma1, sigma2 = np.mean(f1), np.mean(f2), np.std(f1), np.std(f2)  # mean is 0
    bin_scorr = _cross_stat(f1, f2, x, y,
                            report=report, bin_size=bin_size, max_sep=max_sep)
    bin_d = np.arange(0, max_sep + bin_size, bin_size)[:len(bin_scorr)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        bin_s = (bin_scorr - .5 * (mean1 ** 2 + mean2 ** 2)) / sigma1 / sigma2
    return bin_d, bin_s



def fluc_map(x, y, z, bin_size=.03):
    if x.size != y.size or y.size != z.size:
        raise ValueError("The sizes of the three arrays must be equal!")
    x_arr, y_arr, z_arr = x.reshape(-1), y.reshape(-1), z.reshape(-1)
    good = ~np.isnan(x_arr) & ~np.isnan(y_arr) & ~np.isnan(z_arr)
    x_arr, y_arr, z_arr = x_arr[good], y_arr[good], z_arr[good]
    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)
    bin_r, bin_z = _bin_array(r_arr, z_arr, bin_size=bin_size)
    return _step(r_arr, z_arr, bin_r, bin_z)

def inv_where(val_arr, bool_arr, padding=np.nan):
    """
    Recover the val_arr 
    Inverse function of np.where().
    Calculate x = np.where(bool_arr, val_arr, padding),
        when val_arr and bool_arr are known.
    Parameters:
        val_arr: 1D np.array
            The array that has valid values.

        bool_arr: 1D np.array
            The array that has boolean values showing where
                val_arr is valid

        padding: float.
            The value to be padded into the places where
                bool_arr is False
            Defalted to be np.nan.
    
    returns:
        1D np.array
        Padding val_arr using padding following the indication of bool_arr.
    """
    if int(np.sum(bool_arr)) != len(val_arr):
        raise ValueError("The number of 'True' in bool_arr should be equal to " +
                         "the length of val_arr!")
    res = np.ones(len(bool_arr)) * padding
    flag = 0
    for i in range(len(bool_arr)):
        if bool_arr[i]:
            res[i] = val_arr[flag]
            flag += 1
    return res


def deproject(shape, cen_coord=(0, 0), PA=0., b2a=1., q0=0.):
    """
    Deproject the galaxy coordinates using rotation matrix.
    Parameters:
        shape: tuple
            The shape of the original metallicity map from an IFU.

        cen_coord: 2-element tuple
            The coordinates of the galaxy center, shaped as (center_x, center_y)

        PA: float (in unit of degree)
            The position angle. PA = 0 means that the semi long axis is
            aligned to y axis. Thus, adding 90 degrees makes it aligned to x axis.

        q0: float
            A factor related with the intrinsic galaxy disk thickness.
            q0 = 0 means that the disk is infinitely thin.
    
    returns:
        A tuple of (X, Y)
    """
    height, width = shape
    cx, cy = cen_coord
    #  cosi = b2a
    theta = (PA + 90) * np.pi / 180
    dep_mat = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta) / b2a, np.cos(theta) / b2a]])
    x0, y0 = np.meshgrid(range(width), range(height))
    x0, y0 = x0.reshape(-1), y0.reshape(-1)
    xy_mat = np.stack([x0 - cx, y0 - cy], axis=0)
    X, Y = np.dot(dep_mat, xy_mat)
    return X, Y


def corr_func(x_arr, y_arr, met_arr,
              bin_size=.03, max_sep=.6, report=False):
    """
    Compute the two-point correlation function of a deprojected galaxy.
    Parameters:
        x, y, and met: 1D or 2D array
            x and y are the coordinates and met is the metallicities.
            Their sizes must be the same.

        bin_size: float (in unit of kpc)
            It could be as small as the physical spatial resolution.

        max_sep: float (in unit of kpc)
            It is a maximum separation. It is for the sake of saving time
            since a two-point correlation is very close to zero and
            has no information at very large separation.

        report: bool, optional
            If True, then print the procedures and
            how long the two-point correlation function takes.
        
        adp: bool, optional
            If True, then removing the radial metallicity gradient will be
            processed in adaptive bins.
    
    returns:
        sep and ksi: 1D array
        sep is the separation distance, simply an array as
            [0, 1*bin_size, 2*bin_size, ...].
        ksi is the two-point correlation, as
            [1, 0.X, 0.Y, ...].
    """
    if x_arr.size != y_arr.size or y_arr.size != met_arr.size:
        raise ValueError("The sizes of the three arrays must be equal!")
    x, y, met = x_arr.reshape(-1), y_arr.reshape(-1), met_arr.reshape(-1)
    good = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(met)
    x, y, met = x[good], y[good], met[good]
    rad = np.sqrt(x ** 2 + y ** 2)
    bin_rad, bin_met = _bin_array(rad, met, bin_size=bin_size)
    met_fluc = _step(rad, met, bin_rad, bin_met)
    dist, ksi = _tpcf(met_fluc, x, y,
                     report=report, bin_size=bin_size, max_sep=max_sep)
    return dist[dist < max_sep], ksi[dist < max_sep]

def cross_corr_func(x_arr, y_arr, f1_arr, f2_arr,
                    bin_size=.03, max_sep=.6, report=False):
    if x_arr.size != y_arr.size or y_arr.size != f1_arr.size or f1_arr.size != f2_arr.size:
        raise ValueError("The sizes of the four arrays must be equal!")
    x, y, f1, f2 = x_arr.reshape(-1), y_arr.reshape(-1), f1_arr.reshape(-1), f2_arr.reshape(-1)
    good = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(f1) & ~np.isnan(f2)
    x, y, f1, f2 = x[good], y[good], f1[good], f2[good]
    rad = np.sqrt(x ** 2 + y ** 2)
    bin_rad, bin_f1 = _bin_array(rad, f1, bin_size=bin_size)
    f1_fluc = _step(rad, f1, bin_rad, bin_f1)
    bin_rad, bin_f2 = _bin_array(rad, f2, bin_size=bin_size)
    f2_fluc = _step(rad, f2, bin_rad, bin_f2)
    sep, cxy = _ccf(f1_fluc, f2_fluc, x, y,
                    report=report, bin_size=bin_size, max_sep=max_sep)
    return sep[sep < max_sep], cxy[sep < max_sep]





