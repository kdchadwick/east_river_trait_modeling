


import gdal
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import os
import subprocess


parser = argparse.ArgumentParser(description='efficiently extract data from a vector file and multiple accompanying rasters')

parser.add_argument('crown_file',type=str)
parser.add_argument('out_file',type=str)
parser.add_argument('-crown_shape_file',type=str,default=None)
parser.add_argument('-shp_attribute',type=str,default='id')
parser.add_argument('-source_files',nargs='+',type=str)
args = parser.parse_args()

# Open / check all raster files.  Check is very cursory.
file_sets = [gdal.Open(fi,gdal.GA_ReadOnly) for fi in args.source_files]
n_features = 0
for _f in range(len(file_sets)):
    assert file_sets[_f] is not None, 'Invalid input file'
    if (file_sets[_f].RasterXSize != file_sets[0].RasterXSize):
        print('Raster X Size does not match, terminiating')
        quit()
    if (file_sets[_f].RasterYSize != file_sets[0].RasterYSize):
        print('Raster Y Size does not match, terminiating')
        quit()
    n_features += file_sets[_f].RasterCount

if (args.crown_shape_file is not None):
    print('Rasterizing crown shape file')
    if (os.path.isfile(args.crown_file)):
        print('crown_file raster already exists at {}, please remove file to re-rasterize or remove -crown_shape_file argument to use the existing raster'.format(args.crown_file))
        quit()
    trans = file_sets[0].GetGeoTransform()
    cmd_str = 'gdal_rasterize {} {} -a {} -te {} {} {} {} -tr {} {} -init -1'.format(\
              args.crown_shape_file,
              args.crown_file,
              args.shp_attribute,
              trans[0],
              trans[3]+trans[5]*file_sets[0].RasterYSize,
              trans[0]+trans[1]*file_sets[0].RasterXSize,
              trans[3],
              trans[1],
              trans[5])
    print(cmd_str)
    subprocess.call(cmd_str,shell=True)


# Open binary crown file
crown_set = gdal.Open(args.crown_file,gdal.GA_ReadOnly)
crown_trans = crown_set.GetGeoTransform()
assert crown_set is not None, 'Invalid input file'

# Get crown coordinates
crowns = crown_set.ReadAsArray()
crown_coords = np.where(crowns != -1)

# Read through files and grab relevant data
output_array = np.zeros((len(crown_coords[0]),n_features + 3))
for _line in tqdm(range(len(crown_coords[0])),ncols=80):

    output_array[_line,0] = crowns[crown_coords[0][_line], crown_coords[1][_line]]
    output_array[_line,1] = crown_coords[1][_line]*crown_trans[1]+crown_trans[0]
    output_array[_line,2] = crown_coords[0][_line]*crown_trans[5]+crown_trans[3]

    feat_ind = 3
    for _f in range(len(file_sets)):
        line = file_sets[_f].ReadAsArray(0,int(crown_coords[0][_line]),file_sets[_f].RasterXSize,1)
        if (len(line.shape) == 2):
            line = np.reshape(line,(1,line.shape[0],line.shape[1]))

        line = np.squeeze(line[...,crown_coords[1][_line]])

        output_array[_line,feat_ind:feat_ind+file_sets[_f].RasterCount] = line.copy()
        feat_ind += file_sets[_f].RasterCount

# Export
header = ['ID','X_UTM','Y_UTM']
for _f in range(len(file_sets)):
    header.extend([os.path.splitext(os.path.basename(args.source_files[_f]))[0][-4:] +  '_B_' + str(n+1) for n in range(file_sets[_f].RasterCount)])
out_df = pd.DataFrame(data=output_array,columns=header)
out_df.to_csv(args.out_file,sep=',',index=False)

