
# In TYPHOON survey, 1 pixel equals to 1.65 arcsec
arcsec = 4.848e-3  # 1 arcsec = 4.848e-6 rad = 4.848e-3 kpc / Mpc = 4.848 pc / Mpc
arcsec_per_muse_pix = .2

line_rest_wavelength_dict = {
    'OII3727': 3726.032,
    'OII3728': 3727.424,
    'OII3729': 3728.815,
    'Hbeta': 4861.325,
    'OIII4959': 4958.911,
    'OIII5007': 5006.843,
    'NII5755': 5754.590,
    'SIII6312': 6312.060,
    'NII6548': 6548.040,
    'Halpha': 6562.800,
    'NII6584': 6583.460,
    'SII6717': 6716.440,
    'SII6724': 6723.625,
    'SII6731': 6730.810,
    'OII7320': 7319.990,
    'OII7330': 7330.730,
    'SIII9069': 9068.600
}  # From https://astronomy.nmsu.edu/drewski/tableofemissionlines.html

min_T, max_T = 0.6, 2.5      # 10^4 K


