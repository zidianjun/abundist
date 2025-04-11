
from paths import *
from config import eline_dict_muse, diff, min_SN
from constant import min_T, max_T

from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat


def calc_temp(gal_name='N5253', instr='MUSE', save=True):
    eline_data = fits.open(fits_path + gal_name + '_highres_cal_1_comp_WCS_lines.fits')
    eline_dict = eline_dict_muse
    
    SIII6312 = eline_data[eline_dict['SIII6312']].data
    SIII6312e = eline_data[eline_dict['SIII6312'] + diff].data
    SIII9069 = eline_data[eline_dict['SIII9069']].data
    SIII9069e = eline_data[eline_dict['SIII9069'] + diff].data
    SIII6312 = np.where(SIII6312 / SIII6312e > min_SN, SIII6312, np.nan)
    SIII9069 = np.where(SIII9069 / SIII9069e > min_SN, SIII9069, np.nan)

    R_SIII = SIII9069 * 3.44 / SIII6312
    t_SIII = 0.5147 + 0.0003187 * R_SIII + 23.64041 / R_SIII
    t_SIII = np.where((np.nan_to_num(t_SIII) > .6) & (np.nan_to_num(t_SIII) < 2.5),
                t_SIII, np.nan)
    print("%d / %d pixels with T_e estimates in %s" %(
          np.sum(~np.isnan(t_SIII)), t_SIII.size, gal_name))
    t_OIII_i = (t_SIII + 0.32) / 1.19
    t_OII_i = 2.00 / (1. / t_OIII_i + 0.80)
    t_NII_i = 1.85 / (1. / t_OIII_i + 0.72)

    NII5755 = eline_data[eline_dict['NII5755']].data
    NII5755e = eline_data[eline_dict['NII5755'] + diff].data
    NII6584 = eline_data[eline_dict['NII6584']].data
    NII6584e = eline_data[eline_dict['NII6584'] + diff].data
    NII5755 = np.where(NII5755 / NII5755e > min_SN, NII5755, np.nan)
    NII6584 = np.where(NII6584 / NII6584e > min_SN, NII6584, np.nan)
    
    R_NII = NII6584 * 4. / 3. / NII5755
    t_NII = 0.6153 - 0.0001529 * R_NII + 35.3641 / R_NII
    t_NII = np.where((np.nan_to_num(t_NII) > min_T) & (np.nan_to_num(t_NII) < max_T),
                t_NII, np.nan)
    print("%d / %d pixels with T_e estimates in %s" %(
          np.sum(~np.isnan(t_NII)), t_NII.size, gal_name))

    if save:
        np.save(output_path + 'temp/' + gal_name + '_' + instr + '_temp.npy',
                np.stack([t_OII_i, t_NII_i, t_NII, t_SIII, t_OIII_i]))

calc_temp(save=True)




