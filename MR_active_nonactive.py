## Dan Buscombe, Marda Science
## Apr, 2023
##
## Does this:
## 1. 


import json, os
import rioxarray
import xarray as xr 
from glob import glob 
from dask.distributed import Client
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import ndimage
from skimage.exposure import match_histograms
from datetime import datetime

#############################################################
#############################################################
#############################################################
#################### user inputs 

dtype = 'float64'
chunksize = ("auto", "auto")

times = [
    '2012-04-07',
    '2012-08-10',
    '2012-11-08',
    '2013-02-13',
    '2013-04-30',
    '2013-09-19',
    '2014-02-01',
    '2014-09-30',
    '2015-03-03',
    '2015-09-23',
    '2016-01-11',
    '2016-07-14',
    '2016-09-30',
    '2017-09-22'
]

n_workers = 22
threads_per_worker = 2
memory_limit='115GB'

cwd = os.getcwd()

## start client
client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

# Create variable used for time axis
time_var = xr.Variable('time',times)

#############################################################

######### get movie regions and clipper
movie_regions = sorted(glob('../raw_data/GIS/MR*movie*epsg6339.geojson'))
print("{} movie_ regions".format(len(movie_regions)))

movie_geometries = []
for r in movie_regions:
    with open(r) as f:
        gj = json.load(f)
    features = gj['features'][0]

    movie_geometries.append(features['geometry'])

dt = [datetime.strptime(time,'%Y-%m-%d') for time in times]

brfile = '../results/MR/MR_wood/wood_detect/MR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
MRbudget_reaches = gj['features']

MRbudget_reaches_redo = []
for b in MRbudget_reaches:
    MRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))



#############################################################
#########################################################

wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/Elwha_MR_*wood_filtered_bin0.1_regrid_final.tif'))

print(len(wood_files))

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
wood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

#########################################################
## clean up
wood_geotiffs_ds = wood_geotiffs_ds.drop_vars(2)
print(wood_geotiffs_ds.to_array().shape)

# get timeaverage image for consistent lighting
avim_ds = rioxarray.open_rasterio("../results/MR/MR_orthos_orig/Elwha_MR_im_time_mean_prob_regrid.tif", chunks=chunksize, dtype='uint8')
avim_ds = avim_ds.to_dataset('band')
print(avim_ds.dims)
print(wood_geotiffs_ds.dims)

# #########################################
# ################ movies with histogram-matched imagery

#############################################################
im_files = sorted(glob('../raw_data/MR/MR_orthos_orig/Elwha_MR_*_regrid.tif'))
im_files = [i for i in im_files if 'bin' not in i]
print(len(im_files))

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in im_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
im_geotiffs_ds = geotiffs_ds.rename({1: 'red'})
im_geotiffs_ds = im_geotiffs_ds.rename({2: 'green'})
im_geotiffs_ds = im_geotiffs_ds.rename({3: 'blue'})
im_geotiffs_ds = im_geotiffs_ds.drop_vars(4)
print(im_geotiffs_ds.to_array().shape)

### reference image (bright)
reference = im_geotiffs_ds.sel(time='2016-07-14')



## movie_geometries
## wood cumulative sum over dynamic image
for counter,g in tqdm(enumerate(movie_geometries)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)

    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))
    
    wood_da = wood_c.wood.cumsum("time", skipna=True).to_numpy()
    wood_da[wood_da==0] = np.nan

    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.

        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

        fig1, ax1 = plt.subplots()
        ax1.imshow(matched)

        im=ax1.imshow(wood_da[inner_counter,:,:].transpose()/len(times),cmap='inferno', vmin=0, vmax=1)
        plt.colorbar(im, shrink=0.5)
        plt.title(time)
        # plt.show()

        plt.axis('off')
        plt.savefig(f"../results/MR/Wood_inst_movie_{counter}_time_{time}_cumsum.png", dpi=300, bbox_inches='tight')
        plt.close()
    del wood_da, matched, refim_da, im_da




## wood only, relative to start
for counter,g in tqdm(enumerate(movie_geometries)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.

        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

        if inner_counter>0:
            wood_da = wood_c.wood.sel(time=time)
            wood_da = wood_da.transpose().to_numpy()
            wood_da = ndimage.maximum_filter(wood_da, size=10)
            wood_da[wood_da==0] = np.nan
        else:
            wood_da0 = wood_c.wood.sel(time=time)
            wood_da0 = wood_da0.transpose().to_numpy()
            wood_da0 = ndimage.maximum_filter(wood_da0, size=10)
            wood_da0[wood_da0==0] = np.nan        

        if inner_counter>0:
            wood_da = wood_da - wood_da0
        else:
            wood_da = wood_da0.copy()

        fig1, ax1 = plt.subplots()
        ax1.imshow(matched)

        ax1.imshow(wood_da,'inferno')
        plt.title(time)
        # plt.show()

        plt.axis('off')
        plt.savefig(f"../results/MR/Wood_inst_movie_{counter}_time_{time}_rel_to_start.png", dpi=300, bbox_inches='tight')
        plt.close()
        del wood_da





## wood only, relative to previous
for counter,g in tqdm(enumerate(movie_geometries)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.

        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

        wood_da = wood_c.wood.sel(time=time)
        wood_da = wood_da.transpose().to_numpy()
        wood_da[wood_da==0] = np.nan
        
        if inner_counter>0:
            wood_da = wood_da - wood_da0

            wood_da0 = wood_da.copy()
        else:
            wood_da0 = wood_da.copy()

        fig1, ax1 = plt.subplots()
        ax1.imshow(matched)

        im=ax1.imshow(wood_da,cmap='inferno', vmin=-1, vmax=1)
        plt.title(time)
        plt.colorbar(im, shrink=0.5)
        # plt.show()

        plt.axis('off')
        plt.savefig(f"../results/MR/Wood_inst_movie_{counter}_time_{time}_rel_to_previous.png", dpi=300, bbox_inches='tight')
        plt.close()
        del wood_da




for g in tqdm(MRbudget_reaches_redo):


    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
    MR_BR1 = []
    MR_BRm1 = []
    for inner_counter, time in enumerate(times):

        wood_da = wood_c.wood.sel(time=time)
        wood_da = wood_da.transpose().to_numpy()
        wood_da[wood_da==0] = np.nan
        
        if inner_counter>0:
            wood_da = wood_da - wood_da0

            wood_da0 = wood_da.copy()
        else:
            wood_da0 = wood_da.copy()

        result1 = (wood_da0==1).sum().compute().to_numpy()
        MR_BR1.append(float(result1))

        resultm1 = (wood_da0==-1).sum().compute().to_numpy()
        MR_BRm1.append(float(resultm1))

