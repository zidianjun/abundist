
from metcorr import corr_func, cross_corr_func, deproject, fluc_map, inv_where
import diagnostics
from mcmc import fit, KT18_model
import config
import constant
from paths import *

import numpy as np
import pandas as pd
import time
from astropy.io import fits
from os.path import isfile



class DirGal(object):

    def __init__(self, gal_name, instr, diag, map_mode=False, dir_nit=True):
        
        time0 = time.time()
        print("Start analyzing " + gal_name + " with " + diag + " diagnostics in the " +
              instr + " instrument.\n")

        self.name, self.instr = gal_name, instr
        if diag not in ['oxygen', 'nitrogen', 'nitrogen_direct', 'sulphur', 'NO', 'OS', 'NS']:
            raise ValueError("Wrong diagnostics!")
        self.corr_mode = 'auto' if diag.islower() else 'cross'
        if self.corr_mode == 'auto':
            self.diag = 'nitrogen_direct' if diag == 'nitrogen' and dir_nit else diag
        else:
            self.diag = diag[0] + 'd' + diag[1] if diag.startswith('N') and dir_nit else diag
        self.dir_nit = dir_nit
        self.eline_dict = config.eline_dict_muse

        self.eline_data = fits.open(fits_path + gal_name + '_highres_cal_1_comp_WCS_lines.fits')

        self.shape = self.eline_data[0].shape

        Ka, Kb = 3.33, 4.60  # Calzetti et al. (2000)
        self.EBV = 2.5 / (Kb - Ka) * np.log10(
            self.line_flux('Halpha') / self.line_flux('Hbeta') / 2.86)

        self.t_OII_i, self.t_NII_i, self.t_NII, self.t_SIII, self.t_OIII_i = np.load(
            output_path + 'temp/' + gal_name + '_' + self.instr + '_temp.npy')

        par_df = pd.read_csv(obj_path + 'direct_galaxy.csv')
        PA, b2a, distance = par_df[par_df.name == gal_name].iloc[0, 5:8]
        cx, cy = par_df[par_df.name == gal_name].iloc[0, 3:5]

        X, Y = deproject(self.shape, cen_coord=(cx, cy), PA=PA, b2a=b2a)
        self.X, self.Y = X.reshape(self.shape), Y.reshape(self.shape) # X and Y are 1D.

        self.kpc_per_arcsec = distance * constant.arcsec
        arcsec_per_pix = constant.arcsec_per_muse_pix
        self.kpc_per_pix = self.kpc_per_arcsec * arcsec_per_pix
        beam = .9  # Typical MUSE seeing
        self.beam = beam / 2.354 * self.kpc_per_arcsec  # convert to sigma in unit of kpc
        print("One arcsec is %.1f pc. Physical beam size is %.1f pc.\n" %(
              self.kpc_per_arcsec * 1e3, self.beam * 1e3))

        if not map_mode: # If only figures are required, skip the correlation function.
            self.main(err_mode=True)
            if self.corr_mode == 'auto':
                sample_path = (output_path + 'mcmc/' + self.name + '_' + self.diag +
                               '_' + self.instr + '.npy')
                if isfile(sample_path):
                    self.samples = np.load(sample_path)
                    self.par = np.percentile(self.samples, 50, axis=0)
                else:
                    self.samples, self.par = fit(self.dist, self.ksi, self.ksi_u, self.beam)
                    np.save(sample_path, self.samples)
                if diag == 'nitrogen' and dir_nit:
                    self.ksi = np.concatenate([self.ksi[self.dist < .2],
                        self.ksi[self.dist > .2] / self.par[-1]])
                # For illustration purpose. Autocorrelations are not valid >200 pc because
                # of small coverage.

        self.main(err_mode=False)

        time1 = time.time()
        print("Initialization time: %.3fs.\n" %(time1 - time0))

    def main(self, err_mode=True):
        '''
        The parameter err_mode is global.
        If err_mode is True, the loop will run for times,
            and in the eline function Gaussian-generated values will be used.
            This records the 2p correlation and its error.
        If err_mode is False, the loop will only run once,
            and in the eline function only central values will be used.
            This records the original metallicity map.
        '''
        self.err_mode = err_mode

        if self.corr_mode == 'auto':
            corr_func_path = output_path + 'corr_func/' + self.name + '_' + self.diag + '_' + self.instr
            if err_mode and isfile(corr_func_path + '_dist.npy'):
                self.dist = np.load(corr_func_path + '_dist.npy')
                self.ksi = np.load(corr_func_path + '_ksi.npy')
                self.ksi_u = np.load(corr_func_path + '_ksi_u.npy')
                return 0
        else:
            corr_func_path = output_path + 'corr_func/' + self.name + '_' + self.instr
            if err_mode and isfile(corr_func_path + '_c' + self.diag + '.npy'):
                self.sep = np.load(corr_func_path + '_sep.npy')
                self.ksi = np.load(corr_func_path + '_c' + self.diag + '.npy')
                self.ksi_u = np.load(corr_func_path + '_c' + self.diag + '_u.npy')
                return 0

        # The mask changes every time when err_mode is True.

        if self.corr_mode == 'auto':
            valuesXX = None
            for i in range(max(config.times * err_mode, 1)):
                Z, mask = getattr(diagnostics, self.diag)(self)

                if not err_mode:
                    print("Below is the original map.")
                print("%d pixels in realization %d." %(np.sum(mask), i))

                self.Z = np.where(mask, Z, np.nan)
                self.x = np.where(mask, self.X, np.nan) * self.kpc_per_pix
                self.y = np.where(mask, self.Y, np.nan) * self.kpc_per_pix
                
                if err_mode:  # Errors of 2p correlation are only available using bootstrap.
                    dist, ksi = corr_func(self.x, self.y, self.Z, bin_size=self.kpc_per_pix)
                    v = np.expand_dims(ksi, axis=0)
                    valuesXX = np.concatenate([valuesXX, v], axis=0) if valuesXX is not None else v

            if err_mode and valuesXX is not None:  # dist is only available when err_mode is True
                mean, std = valuesXX.mean(0), valuesXX.std(0)
                valid = np.abs(mean) > 1e-4
                # Remove null and keep it shorter to avoid edge effect.
                self.dist = dist[valid]
                self.ksi = mean[valid]
                self.ksi_u = std[valid]
                np.save(corr_func_path + '_dist.npy', self.dist)
                np.save(corr_func_path + '_ksi.npy', self.ksi)
                np.save(corr_func_path + '_ksi_u.npy', self.ksi_u)

        else:
            valuesXY = None
            for i in range(max(config.times * err_mode, 1)):
                if self.diag == 'NO':
                    Z1, m1 = getattr(diagnostics, 'nitrogen' + self.dir_nit * '_direct')(self)
                    Z2, m2 = getattr(diagnostics, 'oxygen')(self)
                elif self.diag == 'OS':
                    Z1, m1 = getattr(diagnostics, 'oxygen')(self)
                    Z2, m2 = getattr(diagnostics, 'sulphur')(self)
                else:
                    Z1, m1 = getattr(diagnostics, 'nitrogen' + self.dir_nit * '_direct')(self)
                    Z2, m2 = getattr(diagnostics, 'sulphur')(self)

                if not err_mode:
                    print("Below is the original map.")
                print("%d and %d pixels in realization %d." %(np.sum(m1), np.sum(m2), i))

                mask_common = m1 & m2
                x = np.where(mask_common, self.X, np.nan) * self.kpc_per_pix
                y = np.where(mask_common, self.Y, np.nan) * self.kpc_per_pix
                Z1 = np.where(mask_common, Z1, np.nan)
                Z2 = np.where(mask_common, Z2, np.nan)
                cutoff = 160
                
                if err_mode:  # Errors of 2p correlation are only available using bootstrap.
                    sep, ksi = cross_corr_func(x, y, Z1, Z2, bin_size=self.kpc_per_pix)
                    v = np.expand_dims(ksi, axis=0)
                    valuesXY = np.concatenate([valuesXY, v], axis=0) if valuesXY is not None else v

            if err_mode and valuesXY is not None:  # dist is only available when err_mode is True
                mean, std = valuesXY.mean(0), valuesXY.std(0)
                valid = np.abs(mean) > 1e-4
                # Remove null and keep it shorter to avoid edge effect.
                self.sep = sep[valid][:cutoff]
                self.ksi = mean[valid][:cutoff]
                self.ksi_u = std[valid][:cutoff]
                np.save(corr_func_path + '_sep.npy', self.sep)
                np.save(corr_func_path + '_c' + self.diag + '.npy', self.ksi)
                np.save(corr_func_path + '_c' + self.diag + '_u.npy', self.ksi_u)

    # ========== line flux and ratio ========== 
    def line_flux(self, label, error=False):
        return self.eline_data[self.eline_dict[label] + config.diff * error].data


    def eline(self, label):
        # First generate a new line flux map and then apply the S/N threshold.
        # If err_mode is True, the mask changes every time.
        f_map = self.eline_data[self.eline_dict[label]].data
        f_map = np.nan_to_num(f_map)
        f_err_map = self.eline_data[self.eline_dict[label] + config.diff].data
        f_err_map = np.where(np.isnan(f_err_map), np.nanmean(f_err_map), f_err_map)
        r_map = np.random.normal(f_map, f_err_map * self.err_mode)
        r_map = np.where(r_map > 0., r_map, 0.)
        signal, noise = r_map.copy(), f_err_map.copy()
        return np.where(signal / noise > config.min_SN, signal, np.nan)        


    def ratio(self, label_up_list, label_down_list, static=False):
        label_list = label_up_list + label_down_list
        reddest_wavelength = 0.
        for label in label_list:
            if constant.line_rest_wavelength_dict[label] > reddest_wavelength:
                reddest_wavelength = constant.line_rest_wavelength_dict[label]
        flux_up, flux_down = 0., 0.

        for label in label_up_list:
            f = self.line_flux(label) if static else self.eline(label)
            flux_up += f * diagnostics.dered_f(self.EBV,
                constant.line_rest_wavelength_dict[label], reddest_wavelength)
        for label in label_down_list:
            f = self.line_flux(label) if static else self.eline(label)
            flux_down += f * diagnostics.dered_f(self.EBV,
                 constant.line_rest_wavelength_dict[label], reddest_wavelength)
        return flux_up / flux_down



