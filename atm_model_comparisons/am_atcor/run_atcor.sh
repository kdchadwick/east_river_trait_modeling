
# These are the code commands used to generate the PLSR models for atmospheric modeling method comparisons

# merge NEON AOP data extractions with trait data, and set up (and run) PLSR models under various conditions
python code/merge_data_run_iterative_plsr.py atm_model_comparisons/am_atcor atm_model_comparisons/am_atcor/CRBU_settings.txt -shade_type dsm -n_folds_to_run 10 -spectra_file data/spectra/atcor_refl_extractions.csv

# run plotting code to create final output plots
python code/plot_10fold_plsr_results.py -folds 10 -results_directory atm_model_comparisons/am_atcor -file_prefix atm_model_comparisons_am_atcor
