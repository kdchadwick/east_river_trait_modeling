import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "Times New Roman"
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.labelpad'] = 6
figure_export_settings = {'dpi': 200, 'bbox_inches': 'tight'}


def main(): 
    df = import_coeffs(models=['am_acorn', 'am_atcor', 'am_isofit'], folds=10)
    #df = import_coeffs(models=['am_atcor'], folds=10)
    plot_coeffs(df)

def import_coeffs(models, folds):
    model_appended = []
    col_names = ["{}{}".format('B',i) for i in range(1,427)]
    col_names = ['Chem', 'Intercept']+col_names

    for _i, i in enumerate(models):
        for f in range(folds):
            # import files
            temp_n = pd.read_csv(os.path.join('atm_model_comparisons', i, 'needles','fold_' + str(f), 'coeff',
                "coeff_values_atm_model_comparisons_"+i+"_"+'needles_fold_' + str(f) + ".csv"), index_col=None, header=None, skiprows=1)
            temp_nn = pd.read_csv(os.path.join('atm_model_comparisons', i, 'noneedles', 'fold_' + str(f), 'coeff',
                "coeff_values_atm_model_comparisons_"+i+"_"+'noneedles_fold_' + str(f) + ".csv"), index_col=None, header=None, skiprows=1)
            temp_nn.columns = col_names
            temp_n.columns = col_names

            # add fold to df
            temp_n['fold']=f
            temp_nn['fold']=f

            # add leaf type to df
            temp_n['leaf_type']='needles'
            temp_nn['leaf_type']='no needles'

            # add model version to df
            temp_n['atm_model']=i
            temp_nn['atm_model']=i

            # append to list
            model_appended.append(temp_n)
            model_appended.append(temp_nn) 
    
    #concatenate list to pandas dataframe for plotting
    all_models=pd.concat(model_appended)
    all_models.to_csv('atm_model_comparisons/output/all_coeffs.csv', index=False)
    return all_models

def plot_coeffs(df, wv_file='data/neon_wavelengths.txt'):
    wv = np.genfromtxt(wv_file)
    color_sets = ['royalblue', 'black', 'forestgreen']
    chems = ['LMA_gm2', 'LWC_per', 'N_weight_percent', 'C_weight_percent', 'CN', 'd13C']
    row_num = int(len(chems)/2)

    for l in ['needles', 'no needles']:
        df_lt = df.loc[df.leaf_type == l]
        figure_base_dir = os.path.join('atm_model_comparisons', 'figs')
        figure_export_settings = {'dpi': 200, 'bbox_inches': 'tight'}
        fig = plt.figure(figsize=(25, 20))
        grid = gridspec.GridSpec(row_num, 2, wspace=.3, hspace=.5)
        for _c, c in enumerate(chems):
            if(_c<row_num):
                ax = plt.subplot(grid[_c, 0])
            else:
                ax = plt.subplot(grid[int(_c-row_num),1])
            # Plot the difference between needles and noneedles in reflectance data
            for _m, m in enumerate(df_lt.atm_model.unique()):
                coeffs = df_lt.loc[(df_lt.atm_model == m) & (df_lt.Chem==c), df_lt.columns.str.startswith('B')]
                
                ax.plot(wv, np.nanmean(coeffs, axis=0), c=color_sets[_m], linewidth=0.5, label=m)
                ax.fill_between(wv, np.nanmean(coeffs, axis=0) - np.nanstd(coeffs, axis=0), np.nanmean(
                    coeffs, axis=0) + np.nanstd(coeffs, axis=0), alpha=.25, facecolor=color_sets[_m])

            if (_c==0): ax.legend()
            ax.set_ylabel('Coefficients')
            ax.set_title(c)

        plt.savefig(os.path.join(figure_base_dir, 'all_coeffs_'+l+'.png'), **figure_export_settings)
        del fig


if __name__ == "__main__":
    main()
