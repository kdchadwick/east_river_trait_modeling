import numpy as np
import gdal
import argparse

import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Apply chem equation to BIL')
    parser.add_argument('refl_dat_f')
    parser.add_argument('output_name')
    parser.add_argument('chem_eq_f_base')
    parser.add_argument('-brightness_normalize', default=True, type=bool)
    args = parser.parse_args()

    intercept_list = []
    slope_list = []
    for fold in range(0, 10):
        chem_dat = np.genfromtxt(args.chem_eq_f_base + str(fold) +
                                 '.csv', delimiter=',', skip_header=1)
        intercept_list.append(chem_dat[:, 1])
        slope_list.append(chem_dat[:, 2:])

    intercept_list = np.stack(intercept_list)
    slope_list = np.stack(slope_list)

    # make sure these match the settings file corresponding to the coefficient file

    # open up raster sets
    dataset = gdal.Open(args.refl_dat_f, gdal.GA_ReadOnly)

    max_y = dataset.RasterYSize
    max_x = dataset.RasterXSize

    # create blank output file
    #driver = gdal.GetDriverByName('ENVI')
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    outDataset = driver.Create(args.output_name,
                               max_x,
                               max_y,
                               slope_list.shape[1]*2,
                               gdal.GDT_Float32)

    outDataset.SetProjection(dataset.GetProjection())
    outDataset.SetGeoTransform(dataset.GetGeoTransform())

    full_bad_bands = np.zeros(426).astype(bool)
    full_bad_bands[:8] = True
    full_bad_bands[192:205] = True
    full_bad_bands[284:327] = True
    full_bad_bands[417:] = True

    # loop through lines [y]
    for l in range(0, max_y):
        dat = np.squeeze(dataset.ReadAsArray(0, l, max_x, 1)).astype(np.float32)
        dat[full_bad_bands, ...] = np.nan

        # do band averaging, which we don't ultimately want - leaving for reference...multiple different versions possible
        #dat = np.nanmean(np.stack([dat[::2,:], dat[1::2,:]]),axis=0)

        if (np.nansum(dat) > 0):
            if (args.brightness_normalize):
                dat = dat / np.sqrt(np.nanmean(np.power(dat, 2), axis=0))[np.newaxis, :]

            dat[np.isnan(dat)] = 0
            for sm in range(0, slope_list.shape[1]):
                output = []
                for fold in range(0, 10):

                    # calculate chem
                    output.append(
                        np.sum(np.multiply(dat, slope_list[fold, sm, :][:, np.newaxis]), axis=0) + intercept_list[fold, sm])
                output_mean = np.mean(output, axis=0)
                outDataset.GetRasterBand(sm+1).WriteArray(output_mean.reshape((1, max_x)), 0, l)

                output_std = np.std(output, axis=0)
                outDataset.GetRasterBand(
                    slope_list.shape[1]+sm+1).WriteArray(output_std.reshape((1, max_x)), 0, l)
                outDataset.FlushCache()

    del outDataset


if __name__ == "__main__":
    main()
