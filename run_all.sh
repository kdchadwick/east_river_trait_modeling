



# run the extraction to get the NEON AOP data coincident with each ground location
# This repository does not hold the specific file references for the below raster files - they can, however, be extracted from google earth engine - see manuscript for details
python extract_aop_data_from_mosaics.py data/crown_raster.tif  data/site_spectra_extracted.csv -crown_shape_file data/CRBU2018_AOP_Crowns_centroid_extended.geojson -source_files ../../mosaic_2019/built_mosaic/min_phase_tch_me ../../mosaic_2019/built_mosaic/min_phase_wtrl ../../mosaic_2019/built_mosaic/min_phase_wtrv ../../mosaic_2019/built_mosaic/min_phase_obs ../../mosaic_2019/built_mosaic/min_phase_shade_centered ../../mosaic_2019/built_mosaic/min_phase_shade_tch ../../mosaic_2019/built_mosaic/min_phase_refl

# merge NEON AOP data extractions with trait data, and set up (and run) PLSR models under various conditions
python merge_data_run_iterative_plsr.py PLSR/test CRBU_settings.txt -shade_type dsm 

# run plotting code to create final output plots
python plot_10fold_plsr_results.py -folds 10 -results_directory PLSR/test
