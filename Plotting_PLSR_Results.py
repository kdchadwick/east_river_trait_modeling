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
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family']= "serif"
plt.rcParams['font.serif']= "Times New Roman"
plt.rcParams['axes.grid']=True
plt.rcParams['axes.axisbelow']=True
plt.rcParams['axes.labelpad']=6
figure_export_settings = {'dpi': 200, 'bbox_inches': 'tight'}


# In[3]:

parser = argparse.ArgumentParser('Plot results from specific east-river trait submodels')
parser.add_argument('-folds', type=str, default=1)
parser.add_argument('-results_directory', default='PLSR_dsm_sg_True')

args = parser.parse_args()


chems = ['N_weight_percent', 'LMA_gm2', 'LWC_per',  'CN', 'C_weight_percent', 'd13C']
chems_names = ['Foliar N (%)', 'Leaf Mass per Area', 'Leaf Water (%)','Foliar C:N Ratio','Foliar C (%)','d13C']
leaftype = ['needles','noneedles']
plotting_colnames = ['SiteID', 'measured', 'modeled', 'calval']


# loop through adding each file and column with the chem name 


for f in range(args.folds):
    chems_results = [pd.DataFrame(columns=['SiteID', 'measured', 'modeled', 'calval', 'chems', 'leaftype'])]
    for n in leaftype:
        for c in chems:
            temp = pd.read_csv(os.path.join(args.results_directory, n, 'fold_' + str(f), 'plot_points', args.results_directory + "_"+n+"_"+'fold_' + str(f) + '_' + c + ".csv"))
            temp.columns = plotting_colnames
            temp['chems'] = c
            temp['leaftype'] = n
            chems_results.append(temp)
    chems_results = pd.concat(chems_results)

    fig = plt.figure(figsize=(19, 13))  # , constrained_layout=True)
    grid = gridspec.GridSpec(2, 3, wspace=.3, hspace=.5)

    r_i = [0, 0, 0, 1, 1, 1]
    c_i = [0, 1, 2, 0, 1, 2]
    for _i in range(len(chems)):
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

        high_val = ((max(max(chems_results.loc[(chems_results['chems'] == i), "measured"]), max(chems_results.loc[(chems_results['chems'] == i), "modeled"]))))
        low_val = ((min(min(chems_results.loc[ (chems_results['chems'] == i), "measured"]), min(chems_results.loc[ (chems_results['chems'] == i), "modeled"]))))
        nrmse = np.sqrt(np.mean(np.power(y-x,2)))/(high_val-low_val)

        ax = plt.subplot(grid[r_i[_i], c_i[_i]])
        ax.plot([low_val, high_val], [low_val, high_val], '--k', alpha=0.5)
        ax.plot(x, intercept + slope * x, 'black', linewidth=2, label='fitted line')
        ax.scatter(chems_results.loc[(chems_results['calval'] == "Calibration") & (chems_results['chems'] == i), "measured"], chems_results.loc[(chems_results['calval'] == "Calibration") & (chems_results['chems'] == i), "modeled"], color="gray", marker='o', alpha=.6)
        ax.scatter(chems_results.loc[(chems_results['calval'] == "Validation") & (chems_results['chems'] == i) & (chems_results['leaftype'] == 'noneedles'), "measured"], chems_results.loc[(chems_results['calval'] == "Validation") & (chems_results['chems'] == i) & (chems_results['leaftype'] == 'noneedles'), "modeled"], color="darkorange", marker='o')
        ax.scatter(chems_results.loc[(chems_results['calval'] == "Validation") & (chems_results['chems'] == i) & (chems_results['leaftype'] == 'needles'), "measured"], chems_results.loc[(chems_results['calval'] == "Validation") & (chems_results['chems'] == i) & (chems_results['leaftype'] == 'needles'), "modeled"], color="royalblue", marker='o')
        ax.set(xlim=[low_val, high_val], ylim=[low_val, high_val], title=i_title, xlabel='Measured', ylabel='Modeled')
        ax.text(low_val, high_val, 'R2: ' + round(r_sq, 2).astype('str') + '   nRMSE: '+ round(nrmse, 2).astype('str'), horizontalalignment='left', verticalalignment='top')

    plt.savefig(os.path.join(os.path.join(args.results_directory, args.results_directory + '_fold_'+ str(f) + '_model_results.png')), **figure_export_settings)
    del fig

# In[14]:


#chemsmod_results = [pd.DataFrame(columns=['SiteID', 'measured', 'modeled', 'calval', 'chems', 'BN', 'leaftype'])]

#for n in leaftype:
#    for b in BN:
#        for c in chems_mod:
#            temp = pd.read_csv("ensemblePLSR/plot_points/" + n + "_" + b + "_10nm_" + c + ".csv")
#            temp.columns = plotting_colnames
#            temp['chems'] = c
#            temp['BN'] = b
#            temp['leaftype'] = n
#            chemsmod_results.append(temp)
#chemsmod_results = pd.concat(chemsmod_results)
#print(chemsmod_results)


# In[6]:


#range(len(chems))


# In[12]:


# In[17]:

#
# fig=plt.figure(figsize=(19,13), constrained_layout=True)
# grid = gridspec.GridSpec(2,3, wspace = .3, hspace = .5)
#
# r_i = [0,0,0,1,1,1]
# c_i = [0,1,2,0,1,2]
# #i = 'N_weight_percent'
# for _i in range(len(chems)):
#     i = chems[_i]
#     i_title = chems_names[_i]
#
#     x = chems_results.loc[(chems_results['calval']=="Validation") & (chems_results['BN']=="BN")& (chems_results['chems']==i),"measured"].to_numpy().reshape(-1, 1)
#     y = chems_results.loc[(chems_results['calval']=="Validation")& (chems_results['BN']=="BN")& (chems_results['chems']==i),"modeled"].to_numpy().reshape(-1, 1)
#     fit_model = LinearRegression()
#     fit_model.fit(x,y)
#     r_sq = fit_model.score(x,y)
#     slope = fit_model.coef_
#     intercept = fit_model.intercept_
#
#     high_val = ((max(max(chems_results.loc[ (chems_results['BN']=="BN")& (chems_results['chems']==i),"measured"]),max(chems_results.loc[(chems_results['BN']=="BN")& (chems_results['chems']==i),"modeled"]))))
#     low_val = ((min(min(chems_results.loc[(chems_results['BN']=="BN")& (chems_results['chems']==i),"measured"]),min(chems_results.loc[(chems_results['BN']=="BN")& (chems_results['chems']==i),"modeled"]))))
#
#     ax = plt.subplot(grid[r_i[_i],c_i[_i]])
#     ax.plot([low_val,high_val],[low_val,high_val],'--k', alpha = 0.5)
#     ax.plot(x, intercept + slope*x, 'black', linewidth=2, label='fitted line')
#     ax.scatter(chems_results.loc[(chems_results['calval']=="Calibration") & (chems_results['BN']=="BN")& (chems_results['chems']==i),"measured"],chems_results.loc[(chems_results['calval']=="Calibration")& (chems_results['BN']=="BN")& (chems_results['chems']==i),"modeled"],color="gray",marker='o',alpha=.6)
#     ax.scatter(chems_results.loc[(chems_results['calval']=="Validation") & (chems_results['BN']=="BN")& (chems_results['chems']==i)& (chems_results['leaftype']=='nn'),"measured"],chems_results.loc[(chems_results['calval']=="Validation")& (chems_results['BN']=="BN")& (chems_results['chems']==i)& (chems_results['leaftype']=='nn'),"modeled"],color="darkorange",marker='o')
#     ax.scatter(chems_results.loc[(chems_results['calval']=="Validation") & (chems_results['BN']=="BN")& (chems_results['chems']==i)& (chems_results['leaftype']=='n'),"measured"],chems_results.loc[(chems_results['calval']=="Validation")& (chems_results['BN']=="BN")& (chems_results['chems']==i)& (chems_results['leaftype']=='n'),"modeled"],color="royalblue",marker='o')
#     ax.set(xlim=[low_val,high_val],ylim=[low_val,high_val], title = i_title, xlabel = 'Measured', ylabel = 'Modeled' )
#     ax.text(low_val, high_val, 'R2: ' + round(r_sq,2).astype('str'),horizontalalignment='left',
#         verticalalignment='top')
#
#
# # In[18]:
#
#
# fig=plt.figure(figsize=(14,35), constrained_layout=True)
# grid = gridspec.GridSpec(5,2)
#
# r_i = [0,0,1,1,2,2,3,3,4,4]
# c_i = [0,1,0,1,0,1,0,1,0,1]
# #i = 'N_weight_percent'
# for _i in range(len(chems_mod)):
#     i = chems_mod[_i]
#
#     x = chemsmod_results.loc[(chemsmod_results['calval']=="Validation") & (chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i),"measured"].to_numpy().reshape(-1, 1)
#     y = chemsmod_results.loc[(chemsmod_results['calval']=="Validation")& (chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i),"modeled"].to_numpy().reshape(-1, 1)
#     fit_model = LinearRegression()
#     fit_model.fit(x,y)
#     r_sq = fit_model.score(x,y)
#     slope = fit_model.coef_
#     intercept = fit_model.intercept_
#
#     high_val = ((max(max(chemsmod_results.loc[ (chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i),"measured"]),max(chemsmod_results.loc[(chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i),"modeled"]))))
#     low_val = ((min(min(chemsmod_results.loc[(chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i),"measured"]),min(chemsmod_results.loc[(chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i),"modeled"]))))
#
#     ax = plt.subplot(grid[r_i[_i],c_i[_i]])
#     ax.plot([low_val,high_val],[low_val,high_val],'--k', alpha = 0.5)
#     ax.plot(x, intercept + slope*x, 'black', linewidth=2, label='fitted line')
#     ax.scatter(chemsmod_results.loc[(chemsmod_results['calval']=="Calibration") & (chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i),"measured"],chemsmod_results.loc[(chemsmod_results['calval']=="Calibration")& (chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i),"modeled"],color="gray",marker='o',alpha=.6)
#     ax.scatter(chemsmod_results.loc[(chemsmod_results['calval']=="Validation") & (chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i)& (chemsmod_results['leaftype']=='nn'),"measured"],chemsmod_results.loc[(chemsmod_results['calval']=="Validation")& (chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i)& (chemsmod_results['leaftype']=='nn'),"modeled"],color="darkorange",marker='o')
#     ax.scatter(chemsmod_results.loc[(chemsmod_results['calval']=="Validation") & (chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i)& (chemsmod_results['leaftype']=='n'),"measured"],chemsmod_results.loc[(chemsmod_results['calval']=="Validation")& (chemsmod_results['BN']=="nBN")& (chemsmod_results['chems']==i)& (chemsmod_results['leaftype']=='n'),"modeled"],color="royalblue",marker='o')
#     ax.set(xlim=[low_val,high_val],ylim=[low_val,high_val], title = i + ' nBN')
#     ax.text(low_val, high_val, 'R2: ' + round(r_sq,2).astype('str'),horizontalalignment='left',
#         verticalalignment='top')
#
#
# # In[72]:
#
#
# fig=plt.figure(figsize=(14,35), constrained_layout=True)
# grid = gridspec.GridSpec(5,2)
#
# r_i = [0,0,1,1,2,2,3,3,4,4]
# c_i = [0,1,0,1,0,1,0,1,0,1]
# #i = 'N_weight_percent'
# for _i in range(len(chems_mod)):
#     i = chems_mod[_i]
#
#     x = chemsmod_results.loc[(chemsmod_results['calval']=="Validation") & (chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i),"measured"].to_numpy().reshape(-1, 1)
#     y = chemsmod_results.loc[(chemsmod_results['calval']=="Validation")& (chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i),"modeled"].to_numpy().reshape(-1, 1)
#     fit_model = LinearRegression()
#     fit_model.fit(x,y)
#     r_sq = fit_model.score(x,y)
#     slope = fit_model.coef_
#     intercept = fit_model.intercept_
#
#     high_val = ((max(max(chemsmod_results.loc[ (chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i),"measured"]),max(chemsmod_results.loc[(chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i),"modeled"]))))
#     low_val = ((min(min(chemsmod_results.loc[(chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i),"measured"]),min(chemsmod_results.loc[(chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i),"modeled"]))))
#
#     ax = plt.subplot(grid[r_i[_i],c_i[_i]])
#     ax.plot([low_val,high_val],[low_val,high_val],'--k', alpha = 0.5)
#     ax.plot(x, intercept + slope*x, 'black', linewidth=2, label='fitted line')
#     ax.scatter(chemsmod_results.loc[(chemsmod_results['calval']=="Calibration") & (chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i),"measured"],chemsmod_results.loc[(chemsmod_results['calval']=="Calibration")& (chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i),"modeled"],color="gray",marker='o',alpha=.6)
#     ax.scatter(chemsmod_results.loc[(chemsmod_results['calval']=="Validation") & (chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i)& (chemsmod_results['leaftype']=='nn'),"measured"],chemsmod_results.loc[(chemsmod_results['calval']=="Validation")& (chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i)& (chemsmod_results['leaftype']=='nn'),"modeled"],color="darkorange",marker='o')
#     ax.scatter(chemsmod_results.loc[(chemsmod_results['calval']=="Validation") & (chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i)& (chemsmod_results['leaftype']=='n'),"measured"],chemsmod_results.loc[(chemsmod_results['calval']=="Validation")& (chemsmod_results['BN']=="BN")& (chemsmod_results['chems']==i)& (chemsmod_results['leaftype']=='n'),"modeled"],color="royalblue",marker='o')
#     ax.set(xlim=[low_val,high_val],ylim=[low_val,high_val], title = i + ' BN')
#     ax.text(low_val, high_val, 'R2: ' + round(r_sq,2).astype('str'),horizontalalignment='left',
#         verticalalignment='top')
#
#
# # In[ ]:
#
#
# #the stats for aggregated models
# fit_model_bn = LinearRegression()
#
# x_bn = BN_2class.loc[BN_2class["3"]=="Validation","1"].to_numpy().reshape(-1, 1)
# y_bn = BN_2class.loc[BN_2class["3"]=="Validation","2"].to_numpy().reshape(-1, 1)
# fit_model_bn.fit(x_bn,y_bn)
#
# r_sq_bn = fit_model_bn.score(x,y)
#
# slope_bn = fit_model_bn.coef_
# intercept_bn = fit_model_bn.intercept_
#
#
# # In[ ]:
#
#
# #the stats for aggregated models
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# fit_model = LinearRegression()
#
# x = noBN_2class.loc[noBN_2class["3"]=="Validation","1"].to_numpy().reshape(-1, 1)
# y = noBN_2class.loc[noBN_2class["3"]=="Validation","2"].to_numpy().reshape(-1, 1)
# fit_model.fit(x,y)
#
# r_sq = fit_model.score(x,y)
#
# slope = fit_model.coef_
# intercept = fit_model.intercept_
#
#
# # In[ ]:
#
#
#
# fig=plt.figure(figsize=(16,5))
#
# ax2 = plt.axes([.425,.1,.25,.8])
# ax2.plot([0,6],[0,6],'--k')
# #ax2.plot(x_bn, intercept_bn + slope_bn*x_bn, 'black', linewidth=2, label='fitted line')
# ax2.scatter(BN_2class.loc[BN_2class["3"]=="Calibration","1"],BN_2class.loc[BN_2class["3"]=="Calibration","2"],color="gray",marker='o',alpha=.6)
# ax2.scatter(BN_con.loc[BN_con["3"]=="Validation","1"],BN_con.loc[BN_con["3"]=="Validation","2"],color="royalblue",marker='o')
# ax2.scatter(BN_nocon.loc[BN_nocon["3"]=="Validation","1"],BN_nocon.loc[BN_nocon["3"]=="Validation","2"],color="darkorange",marker='o')
# ax2.set(title="Brightness Normalized",xlabel='Measured Foliar N (%)',ylabel=('Modeled Foliar N (%)'),xlim=[0,6.5],ylim=[0,6.5], xticks=[0,1,2,3,4,5,6],yticks=[0,1,2,3,4,5,6])
#
# ax3 = plt.axes([.1,.1,.25,.8])
# ax3.plot([0,6.5],[0,6.5],'--k')
# #ax3.plot(x, intercept + slope*x, 'black', linewidth=2, label='fitted line')
# ax3.scatter(noBN_2class.loc[noBN_2class["3"]=="Calibration","1"],noBN_2class.loc[noBN_2class["3"]=="Calibration","2"],color="gray",marker='o',alpha=.6)
# ax3.scatter(noBN_con.loc[noBN_con["3"]=="Validation","1"],noBN_con.loc[noBN_con["3"]=="Validation","2"],color="royalblue",marker='o')
# ax3.scatter(noBN_nocon.loc[noBN_nocon["3"]=="Validation","1"],noBN_nocon.loc[noBN_nocon["3"]=="Validation","2"],color="darkorange",marker='o')
# ax3.set(title="Reflectance",xlabel='Measured Foliar N (%)',ylabel=('Modeled Foliar N (%)'),xlim=[0,6.5],ylim=[0,6.5], xticks=[0,1,2,3,4,5,6],yticks=[0,1,2,3,4,5,6])
#
#
# # In[ ]:
#
#
# #noBN_3class=noBN_broad.append([noBN_con,noBN_mead])
#
#
# # In[ ]:
#
#
# #the stats for aggregated models
# fig=plt.figure(figsize=(16,5))
# #ax1 = plt.axes([.75,.1,.25,.8])
# #ax1.plot([0,6],[0,6],'-k')
# #ax1.scatter(noBN_all.loc[noBN_all["3"]=="Calibration","1"],noBN_all.loc[noBN_all["3"]=="Calibration","2"],color="gray",marker='o',alpha=.6)
# #ax1.scatter(noBN_all.loc[noBN_all["3"]=="Validation","1"],noBN_all.loc[noBN_all["3"]=="Validation","2"],color="red",marker='o')
# #ax1.set(title="All, no BN",xlabel='Measured Foliar N (%)',ylabel=('Modeled Foliar N (%)'),xlim=[0,6],ylim=[0,6], xticks=[0,1,2,3,4,5,6],yticks=[0,1,2,3,4,5,6])
#
# ax2 = plt.axes([.425,.1,.25,.8])
# ax2.plot([0,6],[0,6],'-k')
# ax2.scatter(noBN_2class.loc[noBN_2class["3"]=="Calibration","1"],noBN_2class.loc[noBN_2class["3"]=="Calibration","2"],color="gray",marker='o',alpha=.6)
# ax2.scatter(noBN_2class.loc[noBN_2class["3"]=="Validation","1"],noBN_2class.loc[noBN_2class["3"]=="Validation","2"],color="red",marker='o')
# ax2.set(title="2 Class, no BN",xlabel='Measured Foliar N (%)',ylabel=('Modeled Foliar N (%)'),xlim=[0,6],ylim=[0,6], xticks=[0,1,2,3,4,5,6],yticks=[0,1,2,3,4,5,6])
#
# #ax3 = plt.axes([.1,.1,.25,.8])
# #ax3.plot([0,6],[0,6],'-k')
# #ax3.scatter(noBN_3class.loc[noBN_3class["3"]=="Calibration","1"],noBN_3class.loc[noBN_3class["3"]=="Calibration","2"],color="gray",marker='o',alpha=.6)
# #ax3.scatter(noBN_3class.loc[noBN_3class["3"]=="Validation","1"],noBN_3class.loc[noBN_3class["3"]=="Validation","2"],color="red",marker='o')
# #ax3.set(title="3 Class, no BN",xlabel='Measured Foliar N (%)',ylabel=('Modeled Foliar N (%)'),xlim=[0,6],ylim=[0,6], xticks=[0,1,2,3,4,5,6],yticks=[0,1,2,3,4,5,6])

