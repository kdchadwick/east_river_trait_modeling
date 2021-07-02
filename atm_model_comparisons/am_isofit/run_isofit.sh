
# These are the code commands used to generate the PLSR models for atmospheric modeling method comparisons

# merge NEON AOP data extractions with trait data, and set up (and run) PLSR models under various conditions
python code/merge_data_run_iterative_plsr.py atm_model_comparisons/am_isofit atm_model_comparisons/am_isofit/CRBU_settings.txt -shade_type dsm -n_folds_to_run 10 -spectra_file data/spectra/isofit_extractions_20210204.csv

# run plotting code to create final output plots
python code/plot_10fold_plsr_results.py -folds 10 -results_directory atm_model_comparisons/am_isofit -file_prefix atm_model_comparisons_am_isofit

