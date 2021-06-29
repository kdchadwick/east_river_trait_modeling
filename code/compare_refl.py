






import numpy as np
from osgeo import gdal
import argparse
import os
from spectral.io import envi
import logging

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
plt.switch_backend("Agg")

def main():
    parser = argparse.ArgumentParser(description='Apply chem equation to BIL')
    #parser.add_argument('site')
    parser.add_argument('-file_base', default='../../../mosaics/subsets/')
    parser.add_argument('-wl_file', default='../../../isofit_runs/support/wavelengths.txt')
    args = parser.parse_args()

    bad_bands = np.zeros(426).astype(bool)
    bad_bands[:8] = True
    bad_bands[192:205] = True
    bad_bands[284:327] = True
    bad_bands[417:] = True
    good_bands = np.logical_not(bad_bands)

    wl = np.genfromtxt(args.wl_file)[:,1]*1000

    models = ['atcor','acorn','isofit']
    #sites = ['a','b','c']
    sites = ['a']

    fig = plt.figure(figsize=(6,10))
    gs = gridspec.GridSpec(ncols=len(sites), nrows=12, wspace=0.4,hspace=0.2)

    axes = []
    for _s, site in enumerate(sites):
        print(f'site: {_s}')

        site_axes = []
        cover_ds = envi.open(args.file_base + site + '_cover.hdr')
        cover = cover_ds.open_memmap(writable=False, interleave='bip')[:100,...]
        un_cover = np.unique(cover)
        for _m, model in enumerate(models):
            print(f'model: {_m}')

            path = os.path.join(args.file_base, f'{site}_{model}_refl')
            if os.path.isfile(path + '_scaled.hdr'):
                path += '_scaled.hdr'
            else:
                path += '.hdr'
            refl_ds = envi.open(path)
            refl = refl_ds.open_memmap(writable=False, interleave='bip')[:100,...]

            #if model == 'acorn':
            #    refl = refl / 10000.
            #if model == 'atcor':
            #    refl = refl / 10000.

            if _m == 0:
                color = 'r'
            elif _m == 1:
                color = 'g'
            else:
                color = 'b'

            for _c, c in enumerate(un_cover):
                if _m == 0:
                    ax = fig.add_subplot(gs[_c, _s])
                    site_axes.append(ax)
                else:
                    ax = site_axes[_c]

                subset = np.squeeze(cover == c)

                print(refl.shape)
                subset_refl = np.mean(refl[subset,:],axis=0)

                ax.plot(wl, subset_refl, alpha=0.4, c=color)
                subset_refl[bad_bands] = np.nan
                ax.plot(wl, subset_refl, c=color)

                if _c == len(un_cover) -1:
                    ax.set_xlabel('Wavelength [nm]')
                ax.set_ylim([0,0.55])



    leg_elements = [Line2D([0], [0], color='red', lw=2),
                     Line2D([0], [0], color='green', lw=2),
                     Line2D([0], [0], color='blue', lw=2)]
    plt.legend(leg_elements,models)

    plt.savefig('figs/refl_comp.png',dpi=300,bbox_inches='tight')



            













if __name__ == "__main__":
    main()
