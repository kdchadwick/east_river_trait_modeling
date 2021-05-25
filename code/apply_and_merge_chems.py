import numpy as np
from osgeo import gdal
import argparse
import ray
import multiprocessing
import os
from spectral.io import envi
import logging

import warnings
warnings.filterwarnings('ignore')

def _write_bil_chunk(dat: np.array, outfile: str, line: int, shape: tuple, dtype: str = 'float32') -> None:
    """
    Write a chunk of data to a binary, BIL formatted data cube.
    Args:
        dat: data to write
        outfile: output file to write to
        line: line of the output file to write to
        shape: shape of the output file
        dtype: output data type

    Returns:
        None
    """
    outfile = open(outfile, 'rb+')
    outfile.seek(line * shape[1] * shape[2] * np.dtype(dtype).itemsize)
    outfile.write(dat.astype(dtype).tobytes())
    outfile.close()


@ray.remote
def apply_to_row(args, line_start, line_stop, full_bad_bands, slope_list, intercept_list, slope_list_2 = None, intercept_list_2 = None, balance_raster = None):

    logging.basicConfig(format='%(message)s', level='INFO', filename=args.output_name + 'runlog.txt')
    dataset = envi.open(args.refl_dat_f + '.hdr')
    if balance_raster is not None:
        balanceset = envi.open(balance_raster + '.hdr')

    for line in range(line_start, line_stop):
        #if (line - line_start) % 100 == 0:
            #print('{} : {}/{}'.format(line_start, line-line_start, line_stop-line_start))
        logging.info('{} : {}/{}'.format(line_start, line-line_start, line_stop-line_start))
        dat = dataset.open_memmap(writeable=False, interleave='bip')
        shp = dat.shape

        dat = np.squeeze(dat[line,...]).copy().astype(np.float32)
        dat[...,full_bad_bands] = np.nan

        if balance_raster is not None:
            balance = balanceset.open_memmap(writeable=False,interleave='bip')
            balance = np.squeeze(balance[line,...]).copy().astype(np.float32)

        # do band averaging, which we don't ultimately want - leaving for reference...multiple different versions possible
        #dat = np.nanmean(np.stack([dat[::2,:], dat[1::2,:]]),axis=0)

        if (np.nansum(dat) > 0):
            if (args.brightness_normalize):
                dat = dat / np.sqrt(np.nanmean(np.power(dat, 2), axis=1))[:,np.newaxis]

            dat[np.isnan(dat)] = 0
            outputs = np.zeros((dat.shape[0],slope_list.shape[1]*2))
            for sm in range(0, slope_list.shape[1]):
                output = []
                for fold in range(0, 10):

                    # calculate chem
                    chem_1 = np.sum(np.matmul(dat, slope_list[fold, sm, :][:,np.newaxis]), axis=1) + intercept_list[fold, sm]
                    if slope_list_2 is not None:
                        chem_2 = np.sum(np.matmul(dat, slope_list_2[fold, sm, :][:,np.newaxis]), axis=1) + intercept_list_2[fold, sm]
                        chem = chem_1 * balance + chem_2 * (1-balance)
                    else:
                        chem = chem_1

                    output.append(chem)
                outputs[:,sm] = np.mean(output, axis=0)
                outputs[:,slope_list.shape[1]+sm] = np.std(output, axis=0)


            _write_bil_chunk(np.transpose(outputs), args.output_name, line, (shp[0], outputs.shape[1], shp[1]))




def main():
    parser = argparse.ArgumentParser(description='Apply chem equation to BIL')
    parser.add_argument('refl_dat_f')
    parser.add_argument('output_name')
    parser.add_argument('chem_eq_f_base_1')
    parser.add_argument('-chem_eq_f_base_2', type=str, default=None)
    parser.add_argument('-chem_eq_balance_raster', type=str, default=None)
    parser.add_argument('-brightness_normalize', default=True, type=bool)
    parser.add_argument('-ip_head', type=str)
    parser.add_argument('-n_cores', type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(format='%(message)s', level='INFO', filename=args.output_name + '_runlog.txt')
    if args.n_cores == -1:
        args.n_cores = multiprocessing.cpu_count()

    rayargs = {'address': args.ip_head,
               'local_mode': args.n_cores == 1}
    if args.n_cores < 40:
        rayargs['num_cpus'] = args.n_cores
    ray.init(**rayargs)


    #if os.path.isfile(args.output_name):
    #    quit()

    chem_names = None
    intercept_list = []
    slope_list = []
    for fold in range(0, 10):
        chem_dat = np.genfromtxt(args.chem_eq_f_base_1 + str(fold) +
                                 '.csv', delimiter=',', skip_header=1)
        intercept_list.append(chem_dat[:, 1])
        slope_list.append(chem_dat[:, 2:])

    
    chem_dat = np.genfromtxt(args.chem_eq_f_base_1 + str(fold) +
                             '.csv', delimiter=',', skip_header=1, dtype=str)
    chem_names = chem_dat[:,0]
    print(chem_names)

    intercept_list = np.stack(intercept_list)
    slope_list = np.stack(slope_list)

    if args.chem_eq_f_base_2 is not None:
        intercept_list_2 = []
        slope_list_2 = []
        for fold in range(0, 10):
            chem_dat = np.genfromtxt(args.chem_eq_f_base_2 + str(fold) +
                                     '.csv', delimiter=',', skip_header=1)
            intercept_list_2.append(chem_dat[:, 1])
            slope_list_2.append(chem_dat[:, 2:])

        intercept_list_2 = np.stack(intercept_list_2)
        slope_list_2 = np.stack(slope_list_2)
    else:
        intercept_list_2 = None
        slope_list_2 = None




    # make sure these match the settings file corresponding to the coefficient file

    # open up raster sets
    dataset = gdal.Open(args.refl_dat_f, gdal.GA_ReadOnly)

    max_y = dataset.RasterYSize
    max_x = dataset.RasterXSize

    # create blank output file
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    outDataset = driver.Create(args.output_name,
                               max_x,
                               max_y,
                               slope_list.shape[1]*2,
                               gdal.GDT_Float32,
                               options=['INTERLEAVE=BIL']
                               )

    outDataset.SetProjection(dataset.GetProjection())
    outDataset.SetGeoTransform(dataset.GetGeoTransform())

    for _n, na in enumerate(chem_names):
        outDataset.GetRasterBand(_n+1).SetDescription(na)
        outDataset.GetRasterBand(len(chem_names)+_n+1).SetDescription('std_' + na)
    del outDataset

    full_bad_bands = np.zeros(426).astype(bool)
    full_bad_bands[:8] = True
    full_bad_bands[192:205] = True
    full_bad_bands[284:327] = True
    full_bad_bands[417:] = True

    line_sets = np.linspace(0,max_y,num=args.n_cores,dtype=int)
    logging.info('setting up jobs')
    #print('setting up jobs')
    jobs = [apply_to_row.remote(args, line_sets[_l], line_sets[_l+1], full_bad_bands, slope_list, intercept_list, slope_list_2, intercept_list_2, args.chem_eq_balance_raster) for _l in range(len(line_sets)-1)]


    rreturn = [ray.get(jid) for jid in jobs]
    logging.info('collecting jobs')
    #print('collecting jobs')
    ray.shutdown()

if __name__ == "__main__":
    main()
