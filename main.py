
from config import n_sample, n_walker
from constant import arcsec_per_muse_pix
from dirgal import DirGal
from paths import *
from mcmc import KT18_model

import corner
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


red = '#df1d27'
yellow = '#daa520'
blue = '#367db7'
green = '#4aad4a'
orange = 'darkorange'
purple = 'purple'


def plot_maps():
    gal_name, instr = 'N5253', 'MUSE'
    fig = plt.figure(figsize=(12, 9.6))
    plt.subplots_adjust(left=0.18, bottom=0.10, right=0.86, top=0.95, hspace=0, wspace=0)
    basic_info = pd.read_csv(obj_path + 'direct_galaxy.csv')

    cx, cy = basic_info[basic_info.name == gal_name].iloc[0, 3:5]
    r = int(min(cx, cy) * .95)
    ext = r * arcsec_per_muse_pix

    ax = plt.subplot(221)
    galaxy = DirGal(gal_name, instr, diag='nitrogen', map_mode=True)
    im = ax.imshow(bin_map(galaxy.Z[cy-r:cy+r, cx-r:cx+r]), origin='lower',
        vmin=5.9, vmax=7.1, cmap=plt.cm.RdYlBu_r, extent=[-ext, ext, -ext, ext])
    ax.annotate('12 + log(N/H)', xy=(-28, 24), fontsize=20)
    ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
    ax.errorbar(21, -29, xerr=5, color='k')
    ax.annotate('10"=177 pc', xy=(11, -27), fontsize=15)
    cbar_ax = fig.add_axes([0.03, 0.525, 0.03, 0.425])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('12 + log(N/H)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ax = plt.subplot(222)
    galaxy = DirGal(gal_name, instr, diag='oxygen', map_mode=True)
    im = ax.imshow(bin_map(galaxy.Z[cy-r:cy+r, cx-r:cx+r]), origin='lower',
        vmin=7.7, vmax=8.7, cmap=plt.cm.RdYlBu_r, extent=[-ext, ext, -ext, ext])
    ax.annotate('12 + log(O/H)', xy=(-28, 24), fontsize=20)
    ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
    ax.errorbar(21, -29, xerr=5, color='k')
    ax.annotate('10"=177 pc', xy=(11, -27), fontsize=15)
    cbar_ax = fig.add_axes([0.89, 0.525, 0.03, 0.425])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('12 + log(O/H)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ax = plt.subplot(223)
    im = ax.imshow(bin_map(galaxy.t_SIII[cy-r:cy+r, cx-r:cx+r]), origin='lower',
        vmin=.8, vmax=2.1, cmap=plt.cm.coolwarm, extent=[-ext, ext, -ext, ext])
    ax.annotate('Electron temperature', xy=(-28, 24), fontsize=20)
    ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
    ax.errorbar(21, -29, xerr=5, color='k')
    ax.annotate('10"=177 pc', xy=(11, -27), fontsize=15)
    cbar_ax = fig.add_axes([0.03, 0.10, 0.03, 0.425])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('S$^{++}$ temperature (10$^4$ K)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ax = plt.subplot(224)
    galaxy = DirGal(gal_name, instr, diag='sulphur', map_mode=True)
    im = ax.imshow(bin_map(galaxy.Z[cy-r:cy+r, cx-r:cx+r]), origin='lower',
        vmin=5.9, vmax=7.1, cmap=plt.cm.RdYlBu_r, extent=[-ext, ext, -ext, ext])
    ax.annotate('12 + log(S/H)', xy=(-28, 24), fontsize=20)
    ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
    ax.errorbar(21, -29, xerr=5, color='k')
    ax.annotate('10"=177 pc', xy=(11, -27), fontsize=15)
    cbar_ax = fig.add_axes([0.89, 0.10, 0.03, 0.425])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('12 + log(S/H)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    # plt.show()
    plt.savefig(savefig_path + 'Te_abun_maps.pdf')

def _plot_auto_corr(ax, gal_name, instr, diag, color, norm=False, dir_nit=True):
    galaxy = DirGal(gal_name, instr, diag, dir_nit=dir_nit)
    f_fac = galaxy.par[-1]
    xx = np.arange(0, .6, 1e-3)
    if norm:
        f_cor_ksi = np.concatenate([galaxy.ksi[:1], galaxy.ksi[1:] * f_fac])
        ax.plot(galaxy.dist * 1e3, f_cor_ksi, ls='-', lw=4, color=color, alpha=.5, zorder=1)
        ax.plot(xx * 1e3, KT18_model(xx, *galaxy.par[:3]),
                ls='--', lw=1, color=color, zorder=2)
    else:
        ax.plot(galaxy.dist * 1e3, galaxy.ksi, ls='-', lw=4, color=color, alpha=.5, zorder=1)
        ax.plot(xx * 1e3, np.append([1], KT18_model(xx[1:], *galaxy.par[:3]) / f_fac),
                ls='--', lw=1, color=color, zorder=2)
    ax.fill_between(np.arange(0, galaxy.par[1] * 1e3, .01), -10, 10, color=color, alpha=.2)
    return galaxy


def auto_corr():
    gal_name, instr = 'N5253', 'MUSE'

    plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95, hspace=0)

    ax = plt.subplot2grid((3, 1), (0, 0))
    ax.set_xlim(0, 350)
    ax.set_ylim(-.1, 1.1)
    ax.tick_params(labelbottom=False, labelsize=25)
    ax.set_ylabel("$\\xi_{\mathrm{XX}}(r)$", fontsize=25)
    gO = _plot_auto_corr(ax, gal_name, instr, 'oxygen', blue, norm=False)
    ax.annotate('X = oxygen', color=blue, xy=(.7, .78), xycoords='axes fraction', fontsize=15)
    gN = _plot_auto_corr(ax, gal_name, instr, 'nitrogen', red, norm=False)
    ax.annotate('X = nitrogen', color=red, xy=(.7, .64), xycoords='axes fraction', fontsize=15)
    gS = _plot_auto_corr(ax, gal_name, instr, 'sulphur', yellow, norm=False)
    ax.annotate('X = sulphur', color=yellow, xy=(.7, .5), xycoords='axes fraction', fontsize=15)


    ax = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
    ax.set_xlim(0, 350)
    ax.set_ylim(-.455, 1.45)
    ax.tick_params(labelsize=25)
    ax.set_xlabel("$r$ (pc)", fontsize=25)
    ax.set_ylabel(r"$\Xi_{\mathrm{XX}}(r)$", fontsize=25)
    gO = _plot_auto_corr(ax, gal_name, instr, 'oxygen', blue, norm=True)
    ax.annotate('X = oxygen', color=blue, xy=(.7, .64), xycoords='axes fraction', fontsize=15)
    gN = _plot_auto_corr(ax, gal_name, instr, 'nitrogen', red, norm=True)
    ax.annotate('X = nitrogen', color=red, xy=(.7, .57), xycoords='axes fraction', fontsize=15)
    gS = _plot_auto_corr(ax, gal_name, instr, 'sulphur', yellow, norm=True)
    ax.annotate('X = sulphur', color=yellow, xy=(.7, .50), xycoords='axes fraction', fontsize=15)

    ax.add_patch(Rectangle((0, -.25), 60, .1, fill=False, hatch='xx', color=blue))
    ax.add_patch(Rectangle((0, -.35), 12, .1, fill=False, hatch='xx', color=red))
    ax.add_patch(Rectangle((12, -.35), 48, .1, fill=False, hatch='..', color=red))
    ax.add_patch(Rectangle((0, -.45), 48, .1, fill=False, hatch='xx', color=yellow))
    ax.add_patch(Rectangle((48, -.45), 12, .1, fill=False, hatch='||', color=yellow))
    ax.annotate('Oxygen (100% CC SNe)', color=blue, xy=(63, -.23), fontsize=15)
    ax.annotate('Nitrogen (20% CC SNe + 80% AGB)', color=red, xy=(63, -.33), fontsize=15)
    ax.annotate('Sulphur  (80% CC SNe + 20% Type Ia SNe)', color=yellow, xy=(63, -0.43), fontsize=15)
    ax.annotate('%.1f pc' %(gO.par[1] * 1e3), color=blue,
                xy=(gO.par[1] * 1e3 + 3, .12), fontsize=15)
    ax.annotate('%.1f pc' %(gN.par[1] * 1e3), color=red,
                xy=(gN.par[1] * 1e3 + 3, -.08), fontsize=15)
    ax.annotate('%.1f pc' %(gS.par[1] * 1e3), color=yellow,
                xy=(gS.par[1] * 1e3 + 3, 0.02), fontsize=15)

    # plt.show()
    plt.savefig(savefig_path + 'auto_corr_f_norm.pdf')


def cross_corr():
    gal_name, instr = 'N5253', 'MUSE'

    plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95, hspace=0)
    
    ax = plt.subplot(111)
    ax.set_xlim(-1, 110)
    ax.set_ylim(-.05, .95)
    ax.tick_params(labelsize=25)
    ax.set_xlabel("$r$ (pc)", fontsize=25)
    ax.set_ylabel("$\\xi_{\mathrm{XY}}(r)$", fontsize=25)

    galaxy = DirGal(gal_name, instr, 'NO')
    ax.plot(galaxy.sep * 1e3, galaxy.ksi, lw=3, color=purple, alpha=.5)
    ax.annotate('X = N, Y = O', color=purple, xy=(.7, .66), xycoords='axes fraction', fontsize=15)
    galaxy = DirGal(gal_name, instr, 'OS')
    ax.plot(galaxy.sep * 1e3, galaxy.ksi, lw=3, color=green, alpha=.5)
    ax.annotate('X = O, Y = S', color=green, xy=(.7, .58), xycoords='axes fraction', fontsize=15)
    galaxy = DirGal(gal_name, instr, 'NS')
    ax.plot(galaxy.sep * 1e3, galaxy.ksi, lw=3, color=orange, alpha=.5)
    ax.annotate('X = N, Y = S', color=orange, xy=(.7, .50), xycoords='axes fraction', fontsize=15)

    # plt.show()
    plt.savefig(savefig_path + 'cross_corr.pdf')


def corner_plot():
    gal_name, instr = 'N5253', 'MUSE'
    labels = [r"$w_{\mathrm{inj}}$ (pc)", r"$l$ (pc)", r"$f$"]

    for diag in ['oxygen', 'nitrogen', 'sulphur']:
        samples = np.load(output_path + 'mcmc/' + gal_name + '_' + diag + '_' + instr + '.npy')
        samples = samples.reshape((n_sample * n_walker, 4))
        samples[:, 1] = samples[:, 1] * 1e3
        samples[:, 2] = np.sqrt(samples[:, 2]) * 1e3
        corner.corner(samples[:, 1:], labels=labels, range=[.95] * 3)
        plt.savefig(savefig_path + 'corner_' + gal_name + '_' + diag + '_' +  instr + '.png')

def combine_corner():

    template = Image.open(savefig_path + 'corner_N5253_oxygen_MUSE.png').convert('RGB')
    h, w, c = np.array(template).shape
    res = np.ones((2*h, 2*w, c)) * 255

    img = Image.open(savefig_path + 'corner_N5253_nitrogen_direct_MUSE.png').convert('RGB')
    res[0:h, 0:w, :] = np.array(img)
    img = Image.open(savefig_path + 'corner_N5253_oxygen_MUSE.png').convert('RGB')
    res[0:h, w:2*w, :] = np.array(img)
    img = Image.open(savefig_path + 'corner_N5253_sulphur_MUSE.png').convert('RGB')
    res[h:2*h, w:2*w, :] = np.array(img)

    Image.fromarray((256-res*255).astype(np.uint8)).save(savefig_path + 'corner_N5253_MUSE.png')


def print_perc(diag, dir_nit=True):
    gal_name, instr = 'N5253', 'MUSE'
    galaxy = DirGal(gal_name, instr, diag, dir_nit=dir_nit)
    print(1e3 * (np.percentile(galaxy.samples[:, 1], 50)),
          1e3 * (np.percentile(galaxy.samples[:, 1], 5)),
          1e3 * (np.percentile(galaxy.samples[:, 1], 95)))
    print(1e3 * (np.percentile(np.sqrt(galaxy.samples[:, 2]), 50)),
          1e3 * (np.percentile(np.sqrt(galaxy.samples[:, 2]), 5)),
          1e3 * (np.percentile(np.sqrt(galaxy.samples[:, 2]), 95)))
    print(np.percentile(galaxy.samples[:, 3], 50),
          np.percentile(galaxy.samples[:, 3], 5),
          np.percentile(galaxy.samples[:, 3], 95))

def _bootstrap(cen, std, pdf1, pdf2):
    L = 10000
    res = np.zeros(L)
    for i in range(L):
        p1 = np.random.choice(pdf1)
        p2 = np.random.choice(pdf2)
        res[i] = np.random.normal(cen, std) * np.sqrt(p1 * p2)
    p50, p5, p95 = np.percentile(res, 50), np.percentile(res, 5), np.percentile(res, 95)
    print("%.3f %.3f %.3f\n" %(p50, p5, p95))
    return res


def pdf_cross():
    gal_name, instr = 'N5253', 'MUSE'

    plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.25, bottom=0.16, right=0.95, top=0.96, hspace=0)

    ax = plt.subplot(111)
    pdf_fO = DirGal(gal_name, instr, diag='oxygen').samples[:, -1]
    pdf_fN = DirGal(gal_name, instr, diag='nitrogen').samples[:, -1]
    pdf_fS = DirGal(gal_name, instr, diag='sulphur').samples[:, -1]
    gNO = DirGal(gal_name, instr, diag='NO')
    gOS = DirGal(gal_name, instr, diag='OS')
    gNS = DirGal(gal_name, instr, diag='NS')
    ax.hist(_bootstrap(gNO.ksi[0], gNO.ksi_u[0], pdf_fO, pdf_fN), density=True,
            bins=np.arange(0, 3, .001), color=purple, alpha=.5, histtype='step', lw=2)
    ax.hist(_bootstrap(gOS.ksi[0], gOS.ksi_u[0], pdf_fO, pdf_fS), density=True,
            bins=np.arange(0, 3, .001), color=green,  alpha=.5, histtype='step', lw=2)
    ax.hist(_bootstrap(gNS.ksi[0], gNS.ksi_u[0], pdf_fN, pdf_fS), density=True,
            bins=np.arange(0, 3, .001), color=orange, alpha=.5, histtype='step', lw=2)

    ax.annotate('X = N\nY = O', color=purple, xy=(.16, .9), xycoords='axes fraction', fontsize=15)
    ax.annotate('X = N\nY = S', color=orange, xy=(.02, .9), xycoords='axes fraction', fontsize=15)
    ax.annotate('X = O\nY = S', color=green,  xy=(.87, .9), xycoords='axes fraction', fontsize=15)

    ax.axvline(x=0.95, ls='--', lw=2, color=purple)
    ax.axvline(x=0.99, ls='--', lw=2, color=green)
    ax.axvline(x=0.96, ls='--', lw=2, color=orange)

    ax.set_xlim(.3, 2.1)
    ax.set_ylim(2, 130)
    ax.set_xlabel("$\\Xi_{\mathrm{XY}}(0)$", fontsize=25)
    ax.set_ylabel("Probability\ndensity\nfunction", fontsize=25)
    ax.tick_params(labelsize=25)

    # plt.show()
    plt.savefig(savefig_path + 'pdf_cross.pdf')


def nit_alt(corr):
    gal_name, instr = 'N5253', 'MUSE'

    if corr == 'auto':
        fig = plt.figure(figsize=(15, 5))
        plt.subplots_adjust(left=0.03, bottom=0.22, right=0.96, top=0.98, wspace=.35)

        ax = plt.subplot2grid((1, 3), (0, 0))
        basic_info = pd.read_csv(obj_path + 'direct_galaxy.csv')
        cx, cy = basic_info[basic_info.name == gal_name].iloc[0, 3:5]
        r = int(min(cx, cy) * .95)
        ext = r * arcsec_per_muse_pix

        galaxy = DirGal(gal_name, instr, diag='nitrogen', map_mode=True, dir_nit=False)
        im = ax.imshow(galaxy.Z[cy-r:cy+r, cx-r:cx+r], origin='lower', vmin=5.9, vmax=7.1,
                       cmap=plt.cm.RdYlBu_r, extent=[-ext, ext, -ext, ext])
        ax.annotate("12 + log(N'/H)", xy=(-28, 24), fontsize=20)
        ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
        ax.errorbar(19, -29, xerr=5, color='k')
        ax.annotate('10"=177 pc', xy=(9, -27), fontsize=15)
        cbar_ax = fig.add_axes([0.03, 0.17, 0.252, 0.03])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("12 + log(N'/H)", fontsize=20)
        cbar.ax.tick_params(labelsize=20)

        ax = plt.subplot2grid((1, 3), (0, 1), colspan=2)
        ax.set_xlim(0, 350)
        ax.set_ylim(-.455, 1.45)
        ax.tick_params(labelsize=25)
        ax.set_xlabel("$r$ (pc)", fontsize=25)
        ax.set_ylabel(r"$\Xi_{\mathrm{XX}}(r)$", fontsize=25)
        gO = _plot_auto_corr(ax, gal_name, instr, 'oxygen', blue, norm=True, dir_nit=False)
        ax.annotate('X = oxygen', color=blue, xy=(220, .85), fontsize=15)
        gN = _plot_auto_corr(ax, gal_name, instr, 'nitrogen', red, norm=True, dir_nit=False)
        ax.annotate('X = nitrogen (alternative)', color=red, xy=(220, .7), fontsize=15)
        gS = _plot_auto_corr(ax, gal_name, instr, 'sulphur', yellow, norm=True, dir_nit=False)
        ax.annotate('X = sulphur', color=yellow, xy=(220, .55), fontsize=15)

        ax.add_patch(Rectangle((0, -.25), 60, .1, fill=False, hatch='xx', color=blue))
        ax.add_patch(Rectangle((0, -.35), 12, .1, fill=False, hatch='xx', color=red))
        ax.add_patch(Rectangle((12, -.35), 48, .1, fill=False, hatch='..', color=red))
        ax.add_patch(Rectangle((0, -.45), 48, .1, fill=False, hatch='xx', color=yellow))
        ax.add_patch(Rectangle((48, -.45), 12, .1, fill=False, hatch='||', color=yellow))
        ax.annotate('Oxygen (100% CC SNe)', color=blue, xy=(63, -.23), fontsize=15)
        ax.annotate('Nitrogen (20% CC SNe + 80% AGB)', color=red, xy=(63, -.33), fontsize=15)
        ax.annotate('Sulphur  (80% CC SNe + 20% Type Ia SNe)', color=yellow, xy=(63, -0.43), fontsize=15)
        ax.annotate('%.1f pc' %(gO.par[1] * 1e3), color=blue,
                    xy=(gO.par[1] * 1e3 + 3, .12), fontsize=15)
        ax.annotate('%.1f pc' %(gN.par[1] * 1e3), color=red,
                    xy=(gN.par[1] * 1e3 + 3, -.08), fontsize=15)
        ax.annotate('%.1f pc' %(gS.par[1] * 1e3), color=yellow,
                    xy=(gS.par[1] * 1e3 + 3, 0.02), fontsize=15)

    elif corr == 'cross':
        fig = plt.figure(figsize=(8, 8))
        plt.subplots_adjust(left=0.25, bottom=0.15, right=0.95, top=0.95, hspace=.3)

        ax = plt.subplot(211)
        ax.set_xlim(-1, 110)
        ax.set_ylim(-.05, .95)
        ax.tick_params(labelsize=25)
        ax.set_xlabel("$r$ (pc)", fontsize=25)
        ax.set_ylabel("$\\xi_{\mathrm{XY}}(r)$", fontsize=25)

        galaxy = DirGal(gal_name, instr, 'NO', dir_nit=False)
        ax.plot(galaxy.sep * 1e3, galaxy.ksi, lw=3, color=purple, alpha=.5)
        galaxy = DirGal(gal_name, instr, 'OS', dir_nit=False)
        ax.plot(galaxy.sep * 1e3, galaxy.ksi, lw=3, color=green, alpha=.5)
        galaxy = DirGal(gal_name, instr, 'NS', dir_nit=False)
        ax.plot(galaxy.sep * 1e3, galaxy.ksi, lw=3, color=orange, alpha=.5)

        ax.annotate("X = N', Y = O", color=purple, xy=(.7, .66), xycoords='axes fraction', fontsize=15)
        ax.annotate("X = O, Y = S",  color=green,  xy=(.7, .58), xycoords='axes fraction', fontsize=15)
        ax.annotate("X = N', Y = S", color=orange, xy=(.7, .50), xycoords='axes fraction', fontsize=15)

        ax = plt.subplot(212)
        pdf_fO = DirGal(gal_name, instr, diag='oxygen', dir_nit=False).samples[:, -1]
        pdf_fN = DirGal(gal_name, instr, diag='nitrogen', dir_nit=False).samples[:, -1]
        pdf_fS = DirGal(gal_name, instr, diag='sulphur', dir_nit=False).samples[:, -1]
        gNO = DirGal(gal_name, instr, diag='NO', dir_nit=False)
        gOS = DirGal(gal_name, instr, diag='OS', dir_nit=False)
        gNS = DirGal(gal_name, instr, diag='NS', dir_nit=False)
        ax.hist(_bootstrap(gNO.ksi[0], gNO.ksi_u[0], pdf_fO, pdf_fN), density=True,
                bins=np.arange(0, 3, .001), color=purple, alpha=.5, histtype='step', lw=2)
        ax.hist(_bootstrap(gOS.ksi[0], gOS.ksi_u[0], pdf_fO, pdf_fS), density=True,
                bins=np.arange(0, 3, .001), color=green,  alpha=.5, histtype='step', lw=2)
        ax.hist(_bootstrap(gNS.ksi[0], gNS.ksi_u[0], pdf_fN, pdf_fS), density=True,
                bins=np.arange(0, 3, .001), color=orange, alpha=.5, histtype='step', lw=2)

        ax.annotate("X = N'\nY = O", color=purple, xy=(.16, .8), xycoords='axes fraction', fontsize=15)
        ax.annotate("X = N'\nY = S", color=orange, xy=(.02, .8), xycoords='axes fraction', fontsize=15)
        ax.annotate("X = O\nY = S",  color=green,  xy=(.75, .8), xycoords='axes fraction', fontsize=15)

        ax.axvline(x=0.95, ls='--', lw=2, color=purple)
        ax.axvline(x=0.99, ls='--', lw=2, color=green)
        ax.axvline(x=0.96, ls='--', lw=2, color=orange)

        ax.set_xlim(.3, 2.1)
        ax.set_ylim(2, 130)
        ax.set_xlabel("$\\Xi_{\mathrm{XY}}(0)$", fontsize=25)
        ax.set_ylabel("Probability\ndensity\nfunction", fontsize=25)
        ax.tick_params(labelsize=25)

    else:
        raise ValueError("The parameter 'corr' must be either 'auto' or 'cross'!")

    plt.show()
    # plt.savefig(savefig_path + 'nit_alt_' + corr + '.pdf')




if __name__ == '__main__':

    plot_maps()
    
    auto_corr()
    cross_corr()

    corner_plot()
    combine_corner()
    print_perc('oxygen')
    print_perc('nitrogen')
    print_perc('nitrogen', dir_nit=False)
    print_perc('sulphur')
    
    pdf_cross()

    nit_alt('auto')
    nit_alt('cross')




