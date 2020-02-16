





import pandas as pd
import numpy as np
import subprocess
import os,sys



OBS_DIR = '/lustre/scratch/pbrodrick/colorado/20190607/all_obs/' 
GLT_DIR = '/lustre/scratch/pbrodrick/colorado/20190607/all_glts/' 
IGM_DIR = '/lustre/scratch/pbrodrick/colorado/20190607/all_igm/' 

OUTPUT_DIR = 'predicted_traits'
if (os.path.isdir(OUTPUT_DIR) == False):
  os.mkdir(OUTPUT_DIR)

chem_bases = ['coefficients/coeff_values_PLSR_base_needles_fold_','coefficients/coeff_values_PLSR_base_needles_fold_']

rad_files = np.squeeze(np.array(pd.read_csv('../acorn_atmospheric_correction_2019/all_radiance_files.txt',header=None))).tolist()
refl_files = [os.path.join('../acorn_atmospheric_correction_2019/acorn_mid_variable_vis',os.path.basename(x).replace('rdn','acorn_autovis_refl_ciacorn')) for x in rad_files]
chem_files = [os.path.join(OUTPUT_DIR,os.path.basename(x).replace('rdn','chem_model_')) for x in rad_files]

for _i in range(0,len(rad_files)):
  for _c in range(0,1):
    add_str = 'needles.tif'
    if (_c == 1):
        add_str = 'noneedles.tif'
    cmd_str = 'sbatch -n 1 --mem=15000 -p DGE,SHARED -o logs/o -e logs/e --wrap="export MKL_NUM_THREADS=1; python apply_chems_to_line.py {} {} {}"'.format(refl_files[_i],chem_files[_i] + add_str,chem_bases[_c])
    subprocess.call(cmd_str,shell=True)


