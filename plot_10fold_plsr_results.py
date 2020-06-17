#
# Code for plotting figures from output of PLSR model versions from chem_spectra_merge_svm.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import argparse
import os
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "Times New Roman"
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.labelpad'] = 6
figure_export_settings = {'dpi': 200, 'bbox_inches': 'tight'}


# In[3]:


def main():
    parser = argparse.ArgumentParser('Plot results from specific east-river trait submodels')
    parser.add_argument('-folds', type=int, default=1)
    parser.add_argument('-results_directory', default='PLSR_dsm_sg_True')
    parser.add_argument('-file_prefix', default='PLSR')

    args = parser.parse_args()

    chems = ['N_weight_percent', 'LMA_gm2', 'LWC_per',  'CN', 'C_weight_percent', 'd13C']
    chems_names = ['Foliar N (%)', 'Leaf Mass per Area (g m$^2$)', 'Leaf Water (%)',
                   'Foliar C:N Ratio','Foliar C (%)','$\delta$$^{13}$C']

    leaftype = ['needles', 'noneedles']
    plotting_colnames = ['SiteID', 'measured', 'modeled', 'calval']

    fig = plt.figure(figsize=(19, 13))  # , constrained_layout=True)
    grid = gridspec.GridSpec(2, 3, wspace=.3, hspace=.5)

    r_i = [0, 0, 0, 1, 1, 1]
    c_i = [0, 1, 2, 0, 1, 2]
    round_val = [1, -1, 0, 0, 0, 0]
    for _i in range(len(chems)):

        # get results for this chem
        chems_results = [pd.DataFrame(
            columns=['SiteID', 'measured', 'modeled', 'calval', 'chems', 'leaftype'])]
        for f in range(args.folds):
            for n in leaftype:
                temp = pd.read_csv(os.path.join(args.results_directory, n, 'fold_' + str(f), 'plot_points',
                                                args.file_prefix + "_"+n+"_"+'fold_' + str(f) + '_' + chems[_i] + ".csv"))
                temp.columns = plotting_colnames
                temp['chems'] = chems[_i]
                temp['leaftype'] = n
                chems_results.append(temp)
        chems_results = pd.concat(chems_results)

        i = chems[_i]
        i_title = chems_names[_i]

        x = chems_results.loc[(chems_results['calval'] == "Validation") & (
            chems_results['chems'] == i), "measured"].to_numpy().reshape(-1, 1)
        y = chems_results.loc[(chems_results['calval'] == "Validation") & (
            chems_results['chems'] == i), "modeled"].to_numpy().reshape(-1, 1)
        fit_model = LinearRegression()
        fit_model.fit(x, y)
        r_sq = fit_model.score(x, y)
        slope = fit_model.coef_
        intercept = fit_model.intercept_

        high_val = ((max(max(chems_results.loc[(chems_results['chems'] == i), "measured"]), max(
            chems_results.loc[(chems_results['chems'] == i), "modeled"]))))
        low_val = ((min(min(chems_results.loc[(chems_results['chems'] == i), "measured"]), min(
            chems_results.loc[(chems_results['chems'] == i), "modeled"]))))

        nrmse = np.sqrt(np.mean(np.power(y-x, 2)))/(high_val-low_val)

        high_val += (high_val-low_val)*.05
        low_val -= (high_val-low_val)*.05
        ticks = np.round(np.linspace(low_val, high_val, 4), round_val[_i])
        high_val = ticks[-1]
        low_val = ticks[0]

        ax = plt.subplot(grid[r_i[_i], c_i[_i]])
        ax.plot([low_val, high_val], [low_val, high_val], '--k', alpha=0.5)
        ax.plot(x, intercept + slope * x, 'black', linewidth=2, label='fitted line')

        ax.scatter(chems_results.loc[(chems_results['calval'] == "Validation") & (chems_results['chems'] == i) & (chems_results['leaftype'] == 'noneedles'), "measured"], chems_results.loc[(
            chems_results['calval'] == "Validation") & (chems_results['chems'] == i) & (chems_results['leaftype'] == 'noneedles'), "modeled"], color="darkorange", marker='o')
        ax.scatter(chems_results.loc[(chems_results['calval'] == "Validation") & (chems_results['chems'] == i) & (chems_results['leaftype'] == 'needles'), "measured"], chems_results.loc[(
            chems_results['calval'] == "Validation") & (chems_results['chems'] == i) & (chems_results['leaftype'] == 'needles'), "modeled"], color="royalblue", marker='o')

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set(xlim=[low_val, high_val], ylim=[low_val, high_val],
               title=i_title, xlabel='Measured', ylabel='Modeled')

        ax.text(low_val + (high_val-low_val)*0.02, low_val + (high_val-low_val)*0.98, 'R$^2$: ' + round(r_sq, 2).astype(
            'str') + '   nRMSE: ' + round(nrmse, 2).astype('str'), horizontalalignment='left', verticalalignment='top')
    plt.savefig(os.path.join(os.path.join(args.results_directory, args.file_prefix +
                                          '_allfolds_model_results.png')), **figure_export_settings)
    del fig


if __name__ == "__main__":
    main()
