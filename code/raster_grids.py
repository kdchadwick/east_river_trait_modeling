import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from rasterio.plot import show

raster_path = os.path.join("/Volumes/GoogleDrive/My Drive/CB_share/NEON/comparisons_paper/applied_traits_lma/")

plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "Times New Roman"
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.labelpad'] = 6
figure_export_settings = {'dpi': 200, 'bbox_inches': 'tight'}


models=['acorn', 'atcor', 'isofit']
num_cols=num_rows=len(models)

for i in ['a', 'b', 'c']:

    fig = plt.figure(figsize=(15,15))#*num_rows/(0.5+num_cols*0.7)))
    gs = gridspec.GridSpec(num_cols, num_rows, wspace=0.2,hspace=0.2, width_ratios=[1,1,1.2])

    for c in range(num_cols):
        for r in range(num_rows):
            raster = rasterio.open(os.path.join(raster_path, i + "_atc_" + models[c] + "_model_" + models[r] + ".tif"))
            raster = raster.read(1).astype(float)
            ax = fig.add_subplot(gs[r, c])
            ax.set_title('atc: '+ models[c] + ", model: " + models[r])
            im = ax.imshow(raster, vmin=0, vmax=350, cmap='nipy_spectral')
            if(c==2): cb = fig.colorbar(im, ax=ax, shrink=0.6)
            plt.axis('off')
            
    plt.savefig('atm_model_comparisons/figs/LMA_panel_'+i+'.png',dpi=50,bbox_inches='tight')

