import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from spectral.io import envi
plt.switch_backend('agg')

raster_path = os.path.join("../../../mosaics/subsets/")

plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "Times New Roman"
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.labelpad'] = 6
figure_export_settings = {'dpi': 200, 'bbox_inches': 'tight'}


def read_bil(mm, bands):
    output = np.zeros((mm.shape[0],mm.shape[2],len(bands)))
    for _row in range(mm.shape[0]):
        output[_row,...] = mm[_row,bands,:].T
    return output

def closest(wavelengths, val):
    return np.argsort(np.abs(wavelengths - val))[0]

models=['acorn', 'atcor', 'isofit']
sites = ['a','b','c']
num_cols=len(models)
num_rows=len(sites)

fig = plt.figure(figsize=(15,15))#*num_rows/(0.5+num_cols*0.7)))
gs = gridspec.GridSpec(num_cols, num_rows, wspace=0.2,hspace=0.2, width_ratios=[1,1,1.2])

wl = np.genfromtxt('../../../isofit_runs/support/wavelengths.txt')[:,1]*1000
indices = np.array([closest(wl, 650), closest(wl, 580), closest(wl, 500)])

for r, i in enumerate(sites):

    for c in range(num_cols):
        path = os.path.join(raster_path, f'{i}_{models[c]}_refl')
        if os.path.isfile(path + '_scaled.hdr'):
            path += '_scaled.hdr'
        else:
            path += '.hdr'

        ds = envi.open(path)
        raster = ds.open_memmap(interleave='bil')

        #indices = np.array([0])

        rgb = read_bil(raster,indices)
        print(np.sum(np.all(rgb == 0,axis=-1)))
        print(np.min(rgb),np.max(rgb))
        #rgb = raster[...,indices]

        rgb -= np.percentile(rgb ,2 ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
        rgb /= np.percentile(rgb ,98,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]

        print(r,c)
        ax = fig.add_subplot(gs[r, c])
        ax.set_title(f'{models[c]}-{i}')
        im = ax.imshow(rgb)
        if(c==2): cb = fig.colorbar(im, ax=ax, shrink=0.6)
        plt.axis('off')
            
plt.savefig('figs/rgb.png',dpi=200,bbox_inches='tight')


