

import pandas as pd
import numpy as np
import subprocess
import os
import argparse


def main():

    parser = argparse.ArgumentParser(description='Run conifer predictions on a set of flightlines.')
    parser.add_argument('radiance_file_list', type=str)
    parser.add_argument('reflectance_file_list', type=str)
    parser.add_argument('radiance_file_base', type=str)
    args = parser.parse_args()

    OUTPUT_DIR = 'predicted_traits'
    if (os.path.isdir(OUTPUT_DIR) == False):
        os.mkdir(OUTPUT_DIR)

    chem_bases = ['coefficients/coeff_values_PLSR_base_needles_fold_',
                  'coefficients/coeff_values_PLSR_base_needles_fold_']

    rad_files = np.squeeze(np.array(pd.read_csv(args.radiance_file_list, header=None))).tolist()
    refl_files = [os.path.join(args.radiance_file_base, os.path.basename(
        x).replace('rdn', 'acorn_autovis_refl_ciacorn')) for x in rad_files]
    chem_files = [os.path.join(OUTPUT_DIR, os.path.basename(
        x).replace('rdn', 'chem_model_')) for x in rad_files]

    for _i in range(0, len(rad_files)):
        for _c in range(0, 1):
            add_str = 'needles.tif'
            if (_c == 1):
                add_str = 'noneedles.tif'
            cmd_str = 'sbatch -n 1 --mem=15000 -p DGE,SHARED -o logs/o -e logs/e --wrap="export MKL_NUM_THREADS=1; python apply_chems_to_line.py {} {} {}"'.format(
                refl_files[_i], chem_files[_i] + add_str, chem_bases[_c])
            subprocess.call(cmd_str, shell=True)


if __name__ == "__main__":
    main()
