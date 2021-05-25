import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def main(): 
    import_coeffs(models=['am_acorn', 'am_atcor', 'am_isofit'], folds=10)


def import_coeffs(models, folds):
    all_models = pd.DataFrame()
    for _i, i in enumerate(models):
        model_appended = pd.DataFrame()
        for f in range(10):
            temp_n = pd.read_csv(os.path.join('../atm_model_comparisons', i, 'needles','fold_' + str(f), 'coeff',
                "coeff_values_atm_model_comparisons_"+i+"_"+'needles_fold_' + str(f) + ".csv"))
            temp_n['leaf_type']='needles'
            temp_nn = pd.read_csv(os.path.join('../atm_model_comparisons', i, 'noneedles', 'fold_' + str(f), 'coeff',
                "coeff_values_atm_model_comparisons_"+i+"_"+'noneedles_fold_' + str(f) + ".csv"))
            temp_nn['leaf_type']='no needles'
            model_appended.append(temp_n)
            model_appended.append(temp_nn)
        print(i)
        print(_i)


if __name__ == "__main__":
    main()
