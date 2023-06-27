## Dan Buscombe, Marda Science
## Apr-June, 2023
#

import json, os
import rioxarray
import xarray as xr 
from glob import glob 
import matplotlib.pyplot as plt
import numpy as np
# from dask.distributed import Client
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from area import area
from skimage.measure import label, regionprops_table
from scipy import ndimage
from skimage.exposure import match_histograms

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

# n_workers = 20
# threads_per_worker = 2
# memory_limit='100GB'
#############################################################
## start client
# client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

## factor that converts grid uints 1/8 x 1/8
# into units 1 x 1, i.e. 8 x 8
grid2sqm = 64

# Create variable used for time axis
time_var = xr.Variable('time',times)

dt = [datetime.strptime(time,'%Y-%m-%d') for time in times]

brfile = '../results/LR/LR_wood/wood_detect/model1/LR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
LRbudget_reaches = gj['features']

brfile = '../results/MR/MR_wood/wood_detect/model1/MR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
MRbudget_reaches = gj['features']

LRbudget_reaches_redo = []
for b in LRbudget_reaches:
    LRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))

MRbudget_reaches_redo = []
for b in MRbudget_reaches:
    MRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))


brfile = '../results/LR/LR_wood/wood_detect/model1/LR_budget_reaches_epsg4326.geojson'
with open(brfile) as f:
    gj = json.load(f)
LRbudget_reaches2 = gj['features']

brfile = '../results/MR/MR_wood/wood_detect/model1/MR_budget_reaches_epsg4326.geojson'
with open(brfile) as f:
    gj = json.load(f)
MRbudget_reaches2 = gj['features']

# get area of each budget reach and  put in a list
A_LR = []
for g in tqdm(LRbudget_reaches2):
    A_LR.append(area(g['geometry']))

A_MR = []
for g in tqdm(MRbudget_reaches2):
    A_MR.append(area(g['geometry']))


#############################################################
#############################################################

##########WOOD 
#############################################################
### LR
# get filtered wood probs, clipped to margins
wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/model1/LR_*cleaned.tif'))

print(len(wood_files))

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')

# Rename the variable to a more useful name
LRwood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

### MR
# get filtered wood probs, clipped to margins
wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/model1/MR_*cleaned.tif'))

print(len(wood_files))

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')

# Rename the variable to a more useful name
MRwood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})



################## SEDIMENT
#############################################################
#############################################################
sed_files = sorted(glob('../results/MR/MR_sed/Elwha_*sed.tif'))
print(len(sed_files))

# Load in and concatenate all individual GeoTIFFs for devleopment
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in sed_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
MRsed_geotiffs_ds = geotiffs_ds.rename({1: 'sed'})

print(MRsed_geotiffs_ds.to_array().shape)


sed_files = sorted(glob('../results/LR/LR_sed/Elwha_*sed.tif'))
print(len(sed_files))

# Load in and concatenate all individual GeoTIFFs for devleopment
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in sed_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
LRsed_geotiffs_ds = geotiffs_ds.rename({1: 'sed'})

print(LRsed_geotiffs_ds.to_array().shape)



################## VEGETATION
#############################################################
#############################################################

veg_files = sorted(glob('../raw_data/LR/LR_veg/LR_*_Prob1_regrid.tif'))
# Load in and concatenate all individual GeoTIFFs for vegetation
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in veg_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
LRveg_geotiffs_ds = geotiffs_ds.rename({1: 'veg'})
LRveg_geotiffs_ds = LRveg_geotiffs_ds.drop_vars(2)


veg_files = sorted(glob('../raw_data/MR/MR_veg/MR_*_Prob1_regrid.tif'))
# Load in and concatenate all individual GeoTIFFs for vegetation
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in veg_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
MRveg_geotiffs_ds = geotiffs_ds.rename({1: 'veg'})
MRveg_geotiffs_ds = MRveg_geotiffs_ds.drop_vars(2)


#############################################################
#############################################################

### merged MR
for time in times:
    sed_da = MRsed_geotiffs_ds.sed.sel(time=time)
    wood_da = MRwood_geotiffs_ds.wood.sel(time=time)
    veg_da = MRveg_geotiffs_ds.veg.sel(time=time)
    dummy_da = MRveg_geotiffs_ds.veg.sel(time=time)

    mask = veg_da==0

    veg_da = ndimage.maximum_filter(veg_da, size=10)
    veg_da[veg_da<.5] = np.nan  
    veg_da[veg_da>0]=1

    sed_da = ndimage.maximum_filter(sed_da, size=10)
    sed_da[sed_da<.5] = np.nan       
    sed_da[sed_da>0]=1

    wood_da = ndimage.maximum_filter(wood_da, size=10)
    wood_da[wood_da==0] = np.nan
    wood_da[wood_da>0]=1

    water_da = np.ones_like(wood_da, dtype=np.float32)
    water_da[wood_da==1] = np.nan
    water_da[sed_da==1] = np.nan
    water_da[veg_da==1] = np.nan
    water_da[mask==1] = np.nan

    out_da = np.zeros_like(wood_da, dtype=np.float32)
    out_da[water_da==1] = 1
    out_da[sed_da==1] = 2
    out_da[veg_da==1] = 3
    out_da[wood_da==1] = 4

    dummy_da.data = out_da
    dummy_da.rio.to_raster(raster_path=f"../results/MR/MR_all/Elwha_{time}_all_est.tif", compress='zstd', zstd_level=1, num_threads='all_cpus', tiled=True, dtype='uint8', driver="GTiff", predictor=2, windowed=True)

    del dummy_da, out_da, water_da, wood_da, sed_da, veg_da, mask



### merged LR
for time in times:
    sed_da = LRsed_geotiffs_ds.sed.sel(time=time)
    wood_da = LRwood_geotiffs_ds.wood.sel(time=time)
    veg_da = LRveg_geotiffs_ds.veg.sel(time=time)
    dummy_da = LRveg_geotiffs_ds.veg.sel(time=time)

    mask = veg_da==0

    veg_da = ndimage.maximum_filter(veg_da, size=10)
    veg_da[veg_da<.5] = np.nan  
    veg_da[veg_da>0]=1

    sed_da = ndimage.maximum_filter(sed_da, size=10)
    sed_da[sed_da<.5] = np.nan       
    sed_da[sed_da>0]=1

    wood_da = ndimage.maximum_filter(wood_da, size=10)
    wood_da[wood_da==0] = np.nan
    wood_da[wood_da>0]=1

    water_da = np.ones_like(wood_da, dtype=np.float32)
    water_da[wood_da==1] = np.nan
    water_da[sed_da==1] = np.nan
    water_da[veg_da==1] = np.nan
    water_da[mask==1] = np.nan


    out_da = np.zeros_like(wood_da, dtype=np.float32)
    out_da[water_da==1] = 1
    out_da[sed_da==1] = 2
    out_da[veg_da==1] = 3
    out_da[wood_da==1] = 4

    dummy_da.data = out_da
    dummy_da.rio.to_raster(raster_path=f"../results/LR/LR_all/Elwha_{time}_all_est.tif", compress='zstd', zstd_level=1, num_threads='all_cpus', tiled=True, dtype='uint8', driver="GTiff", predictor=2, windowed=True)

    del dummy_da, out_da, water_da, wood_da, sed_da, veg_da, mask    