
import sys
import logging
from scipy.signal import savgol_filter
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')


def main():
    parser = argparse.ArgumentParser('Run specific east-river trait submodels')
    parser.add_argument('output_directory', type=str)
    parser.add_argument('base_settings_file', type=str)
    parser.add_argument('-wavelength_file', default='data/neon_wavelengths.txt')
    parser.add_argument('-shade_type', type=str,
                        choices=['dsm', 'tch', 'both', 'none'], default=None)
    parser.add_argument('-chem_file', default='data/site_trait_data.csv')
    parser.add_argument('-spectra_file', default='data/spectra/20200214_CRBU2018_AOP_Crowns_extraction.csv')
    parser.add_argument('-bn', '--brightness_normalize', type=str, default='True')
    parser.add_argument('-sn', '--spectrally_normalize', type=str, default='False')
    parser.add_argument('-spectral_smoothing',
                        choices=['sg', 'none', '2band', '3band'], type=str, default='none')
    parser.add_argument('-n_test_folds', default=10, type=int)
    parser.add_argument('-n_folds_to_run', default=1, type=int)

    parser.add_argument('-ndvi_min', default=0.5, type=float)

    parser.add_argument('-plsr_ensemble_code_dir',
                        default='/Users/kdchadwick/Github/crown_based_ensembling', type=str)

    args = parser.parse_args()
    if args.brightness_normalize.lower() == 'true':
        args.brightness_normalize = True
    else:
        args.brightness_normalize = False
        
    if args.spectrally_normalize.lower() == 'true':
        args.spectrally_normalize = True
    else:
        args.spectrally_normalize = False

    assert args.n_folds_to_run <= args.n_test_folds, 'n_folds_to_run <= n_test_folds'

    #assert os.path.isdir(args.output_directory) is False, 'output directory already exists'
    subprocess.call('mkdir ' + os.path.join(args.output_directory), shell=True)

    logging.basicConfig(filename=os.path.join(args.output_directory, 'log_file.txt'), level='INFO')

    sys.path.append(args.plsr_ensemble_code_dir)
    import read_settings_file
    settings_file = read_settings_file.settings(args.base_settings_file)

    def find_nearest(array_like, v):
        index = np.argmin(np.abs(np.array(array_like) - v))
        return index

    plt.rcParams.update({'font.size': 20})
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['font.serif'] = "Times New Roman"
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.labelpad'] = 6

    # For jupyter notebooks
    CN = pd.read_csv(args.chem_file)
    extract = pd.read_csv(args.spectra_file)

    # saving column names
    headerCN = list(CN)
    headerSpec = list(extract)  # Reflectance bands start at 18 (17 in zero base)
    # logging.info(headerSpec)

    # removing shaded pixels
    if args.shade_type == 'dsm':
        extract = extract.loc[extract['ered_B_1'] == 1]
    elif args.shade_type == 'tch':
        extract = extract.loc[extract['_tch_B_1'] == 1]
    elif args.shade_type == 'both':
        extract = extract.loc[(extract['_tch_B_1'] == 1) & (extract['ered_B_1'] == 1)]
    elif args.shade_type == 'none':
        extract = extract.loc[extract['_tch_B_1'] != -9999]

    # calculating NDVI using bands 54 and 96 for non-meadow sites
    NDVI = (np.array(extract[settings_file.get_setting('band preface') + "96"]) -
            np.array(extract[settings_file.get_setting('band preface') + "54"])) / \
        (np.array(extract[settings_file.get_setting('band preface') + "96"]) +
         np.array(extract[settings_file.get_setting('band preface') + "54"]))
    NDVImask = NDVI < args.ndvi_min
    extract = extract.drop(extract[NDVImask].index)

    CNmeadows = CN.loc[CN['Site_Veg'] == 'Meadow']
    CNconifer = CN.loc[CN['Needles'] == True]
    CNbroadleaf = CN.loc[(CN['Needles'] != True) & (CN['Site_Veg'] != 'Meadow')]

    np.random.seed(6)

    fraction_sets = [1./float(args.n_test_folds) for i in range(args.n_test_folds)]

    CalValM = np.random.choice(list(range(args.n_test_folds)),
                               size=(CNmeadows.shape[0],), p=fraction_sets)
    CalValC = np.random.choice(list(range(args.n_test_folds)),
                               size=(CNconifer.shape[0],), p=fraction_sets)
    CalValB = np.random.choice(list(range(args.n_test_folds)),
                               size=(CNbroadleaf.shape[0],), p=fraction_sets)

    CNmeadows['CalVal'] = CalValM
    CNconifer['CalVal'] = CalValC
    CNbroadleaf['CalVal'] = CalValB

    # Merging chem data with the correct extraction data based on the
    conifers = pd.merge(CNconifer, extract, how='inner', right_on=['ID'], left_on=['SampleSiteID'])
    meadows = pd.merge(CNmeadows, extract, how='inner', right_on=['ID'], left_on=['SampleSiteID'])
    broadleaf = pd.merge(CNbroadleaf, extract, how='inner',
                         right_on=['ID'], left_on=['SampleSiteID'])

    # Concatenating data for different subset exports
    noneedles = meadows.append(broadleaf)

    # first column of reflectance data
    rfdat = list(extract).index(settings_file.get_setting('band preface') + '1')

    # defining wavelengths
    wv = np.genfromtxt(args.wavelength_file)
    bad_bands = []
    good_band_ranges = []

    bad_band_ranges = [[0, 425], [1345, 1410], [1805, 2020], [2470, 2700]]
    for _bbr in range(len(bad_band_ranges)):
        bad_band_ranges[_bbr] = [find_nearest(wv, x) for x in bad_band_ranges[_bbr]]
        if (_bbr > 0):
            good_band_ranges.append([bad_band_ranges[_bbr-1][1], bad_band_ranges[_bbr][0]])

        for n in range(bad_band_ranges[_bbr][0], bad_band_ranges[_bbr][1]):
            bad_bands.append(n)
    bad_bands.append(len(wv)-1)

    good_bands = np.array([x for x in range(0, 426) if x not in bad_bands])

    all_band_indices = (np.array(good_bands)+rfdat).tolist()
    all_band_indices.extend((np.array(bad_bands)+rfdat).tolist())
    all_band_indices = np.sort(all_band_indices)

    # extracting needle and non-needle samples
    conifer_spectra = np.array(conifers[np.array(headerSpec)[all_band_indices]]).astype(np.float32)
    conifer_spectra[:, bad_bands] = np.nan

    noneedles_spectra = np.array(noneedles[np.array(headerSpec)[all_band_indices]]).astype(np.float32)
    noneedles_spectra[:, bad_bands] = np.nan

    spectra_sets = [conifer_spectra, noneedles_spectra]
    color_sets = ['royalblue', 'darkorange']

    #####   Spectral smoothing #####
    if (args.spectral_smoothing == '2band' or args.spectral_smoothing == '3band'):
        average_interval = 2
        if args.spectral_smoothing == '3band':
            average_interval = 3
        for _s in range(len(spectra_sets)):
            spectra = spectra_sets[_s]
            av_spec = [spectra[:, ::average_interval]]
            if _s == 0:
                smoothed_spectra_wavelengths = 1 / float(average_interval) * wv[::average_interval]
            for _i in range(1, average_interval):
                av_spec.append(spectra[:, _i::average_interval])
                if _s == 0:
                    smoothed_spectra_wavelengths += 1 / \
                        float(average_interval) * wv[_i::average_interval]

            av_spec = np.stack(av_spec)
            av_spec = np.nanmean(av_spec, axis=0)
            spectra_sets[_s] = av_spec
            if (_s == 0):
                wv = smoothed_spectra_wavelengths

    elif (args.spectral_smoothing == 'sg'):
        for _s in range(len(spectra_sets)):
            spectra = spectra_sets[_s]
            for _gbr in range(len(good_band_ranges)):
                spectra[:, good_band_ranges[_gbr][0]:good_band_ranges[_gbr][1]] = \
                    savgol_filter(
                        spectra[:, good_band_ranges[_gbr][0]:good_band_ranges[_gbr][1]], window_length=5, polyorder=3, axis=1)
            spectra_sets[_s] = spectra

    bad_bands = np.where(np.any(np.isnan(spectra_sets[0]), axis=0))[0].tolist()

    if args.brightness_normalize:
        for _s in range(len(spectra_sets)):
            spectra = spectra_sets[_s]
            spectra = spectra / np.sqrt(np.nanmean(np.power(spectra, 2), axis=1))[:, np.newaxis]
            spectra_sets[_s] = spectra
            
    if args.spectrally_normalize:
        for _s in range(len(spectra_sets)):
            spectra = spectra_sets[_s]
            spectra -= np.nanmean(spectra, axis=1)[:,np.newaxis]
            spectra /= np.nanstd(spectra, axis=1)[:,np.newaxis]
            spectra_sets[_s] = spectra

    ############### Rebuild dataframes for export  ##############
    export_dataframes = [conifers.copy(), noneedles.copy()]
    for _s in range(len(spectra_sets)):
        for b in range(spectra_sets[_s].shape[1] + 1, 427):
            export_dataframes[_s] = export_dataframes[_s].drop('refl_B_{}'.format(b), axis=1)
        for b in range(spectra_sets[_s].shape[1]):
            export_dataframes[_s]['refl_B_{}'.format(b + 1)] = spectra_sets[_s][:, b]

    conifers_df_export = export_dataframes[0]
    noneedles_df_export = export_dataframes[1]
    aggregated_df_export = conifers_df_export.append(noneedles_df_export)

    # Export to output files
    subprocess.call('mkdir ' + os.path.join(args.output_directory, 'data'), shell=True)
    output_df_set_files = []
    output_df_set_files.append(os.path.join(args.output_directory, 'data', 'extraction_chem.csv'))
    output_df_set_files.append(os.path.join(args.output_directory,
                                            'data', 'extraction_chem_needles.csv'))
    output_df_set_files.append(os.path.join(args.output_directory,
                                            'data', 'extraction_chem_noneedles.csv'))

    aggregated_df_export.to_csv(output_df_set_files[0], index=False, sep=',')
    conifers_df_export.to_csv(output_df_set_files[1], index=False, sep=',')
    noneedles_df_export.to_csv(output_df_set_files[2], index=False, sep=',')

    # Plot the difference between needles and noneedles in reflectance data
    figure_base_dir = os.path.join(args.output_directory, 'figures')
    subprocess.call('mkdir ' + figure_base_dir, shell=True)
    figure_export_settings = {'dpi': 200, 'bbox_inches': 'tight'}

    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    for _s in range(len(spectra_sets)):
        spectra = spectra_sets[_s]
        plt.plot(wv, np.nanmean(spectra, axis=0), c=color_sets[_s], linewidth=2)
        plt.fill_between(wv, np.nanmean(spectra, axis=0)  - np.nanstd(spectra, axis=0) , np.nanmean(
            spectra, axis=0) + np.nanstd(spectra, axis=0), alpha=.35, facecolor=color_sets[_s])

    plt.legend(['Needle', 'Non-Needle'])
    plt.ylabel('Reflectance (%)')
    if args.brightness_normalize:
        plt.ylabel('Brightness Norm. Reflectance')
    elif args.spectrally_normalize:
        plt.ylabel('Spectrally Norm. Reflectance')
    else:
        plt.ylabel('Reflectance (%)')
    plt.xlabel('Wavelength (nm)')

    plt.savefig(os.path.join(figure_base_dir, 'class_spectra.png'), **figure_export_settings)
    del fig

    # Run through and generate the PLSR settings files, and call the PLSR code
    starting_dir = os.getcwd()
    for output_df_set in output_df_set_files:

        chem_output_dir = os.path.splitext(os.path.basename(output_df_set))[0]
        chem_output_dir = chem_output_dir.split('_')[-1]
        chem_output_dir = os.path.join(args.output_directory, chem_output_dir)
        subprocess.call('mkdir ' + chem_output_dir, shell=True)
        for fold in range(args.n_folds_to_run):

            bad_bands_str = ''
            for n in bad_bands:
                bad_bands_str += str(n+1) + ','
            settings_file.settings_obj['Spectral']['bad bands(1-based)'] = bad_bands_str[:-1]

            bn_norm_str = 'False'
            settings_file.settings_obj['Spectral']['brightness normalize'] = bn_norm_str
            
            pw_norm_str = 'False'
            settings_file.settings_obj['Spectral']['pixel-wise scaling'] = pw_norm_str

            settings_file.settings_obj['Data']['csv file'] = os.path.join(
                os.getcwd(), output_df_set)

            settings_file.settings_obj['Data']['test set value'] = str(fold)

            if args.spectral_smoothing == '2band':
                settings_file.settings_obj['Spectral']['max band'] = '213'
                settings_file.settings_obj['Spectral']['wavelength interval(nm)'] = '10'
            elif args.spectral_smoothing == '3band':
                settings_file.settings_obj['Spectral']['max band'] = '142'
                settings_file.settings_obj['Spectral']['wavelength interval(nm)'] = '15'

            fold_output_dir = os.path.join(chem_output_dir, 'fold_' + str(fold))
            subprocess.call('mkdir ' + fold_output_dir, shell=True)

            settings_file.settings_obj['General']['version name'] = fold_output_dir.replace(
                '/', '_')

            output_sf = os.path.join(os.getcwd(), os.path.join(
                fold_output_dir, 'settings_file.txt'))
            with open(output_sf, 'w') as configfile:
                settings_file.settings_obj.write(configfile)

            os.chdir(fold_output_dir)
            cmd_str = 'python {} {}&'.format(os.path.join(
                args.plsr_ensemble_code_dir, 'ensemble_plsr.py'), output_sf)
            logging.info('calling:\n{}'.format(cmd_str))
            subprocess.call(cmd_str, shell=True)
            os.chdir(starting_dir)


if __name__ == "__main__":
    main()
