## Dan Buscombe, Marda Science
## Apr, 2023
##
## Does this:
## 1. calls bash scripts to use gdal to clip rasters to channel margins
## 2. then regrid to a common extent and pixel size
## 3. then make a datacube of Elwha MR wood, veg, water, dev
## 4. then create time-averages of veg, water, dev
## 5. then filter wood by time-averages of veg, water, dev
## 6. then chunk filtered wood by region

## Where are we in the sequence?
## 1. >>>> filter_wood_by_av_veg_water_dev.py
## 2. make_wood_movies.py
## 3. timeseries_analysis.py

import json, os
import rioxarray
import xarray as xr 
from glob import glob 
from dask.distributed import Client
from tqdm import tqdm

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

n_workers = 20
threads_per_worker = 2
memory_limit='100GB'

cwd = os.getcwd()

run_bash = True

#############################################################

### recombine (mosaic) and regrid
# all "results" rasters are 15928 x 41411
# pixel = 1.569605128802169152e-06 degrees (approx 15cm)
# gridded to extents of grid.geojson

if run_bash:
    os.chdir("../raw_data/MR/MR_orthos_orig/")
    os.system("bash regrid.sh")
    os.chdir(cwd)

    os.chdir("../raw_data/MR/MR_wood/")
    ## clip to channel margins
    os.system("../raw_data/bash regrid_cm.sh")
    # regrid to common extent and pixel size
    os.system("../raw_data/bash regrid.sh")
    os.chdir(cwd)

    os.chdir("../raw_data/MR/MR_dev/")
    os.system("bash regrid_cm.sh")
    os.system("bash regrid.sh")
    os.chdir(cwd)

    os.chdir("../raw_data/MR/MR_water/")
    os.system("bash regrid_cm.sh")
    os.system("bash regrid.sh")
    os.chdir(cwd)

    os.chdir("../raw_data/MR/MR_veg/")
    os.system("bash regrid_cm.sh")
    os.system("bash regrid.sh")
    os.chdir(cwd)

    os.chdir("../raw_data/MR/MR_all/")
    os.system("bash regrid_cm.sh")
    os.system("bash regrid.sh")
    os.system("bash extract_sed_grids.sh")
    os.chdir(cwd)

## read regridded "Prob" mosaic files
allclass_files = sorted(glob('../raw_data/MR/MR_all/MR_*_regrid.tif'))
wood_files = sorted(glob('../raw_data/MR/MR_wood/MR_*_Prob1_regrid.tif'))
veg_files = sorted(glob('../raw_data/MR/MR_veg/MR_*_Prob1_regrid.tif'))
water_files = sorted(glob('../raw_data/MR/MR_water/MR_*_Prob0_regrid.tif'))
dev_files = sorted(glob('../raw_data/MR/MR_dev/MR_*_Prob1_regrid.tif'))
im_files = sorted(glob('../raw_data/MR/MR_orthos_orig/Elwha_MR_*_regrid.tif'))
im_files = [i for i in im_files if 'bin' not in i]

print(len(wood_files))
print(len(veg_files))
print(len(water_files))
print(len(dev_files))
print(len(im_files))
print(len(times))
print(len(allclass_files))

######### get regions 
regions = sorted(glob('../GIS/MR*ID*.geojson'))
regions = [r for r in regions if 'pts' not in r]
print("{} regions".format(len(regions)))

#################### set up dask session

## start client
client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

# Create variable used for time axis
time_var = xr.Variable('time',times)

geometries = []
for r in regions:
    with open(r) as f:
        gj = json.load(f)
    features = gj['features'][0]

    geometries.append(features['geometry'])

#############################################################
#############################################################
#############################################################
## we start with folders of wood, veg, dev probablities, as well as images
## we load them in, then save out as geotiff regions

#############################################################
# Load in and concatenate all individual GeoTIFFs for water
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in water_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
water_geotiffs_ds = geotiffs_ds.rename({1: 'water'})

#############################################################
# Load in and concatenate all individual GeoTIFFs for vegetation
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in veg_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
veg_geotiffs_ds = geotiffs_ds.rename({1: 'veg'})

#############################################################
# Load in and concatenate all individual GeoTIFFs for devleopment
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dev_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
dev_geotiffs_ds = geotiffs_ds.rename({1: 'dev'})

#############################################################
# Load in and concatenate all individual GeoTIFFs for ortho images
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in im_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
im_geotiffs_ds = geotiffs_ds.rename({1: 'red'})
im_geotiffs_ds = im_geotiffs_ds.rename({2: 'green'})
im_geotiffs_ds = im_geotiffs_ds.rename({3: 'blue'})

#############################################################
## clean up
water_geotiffs_ds = water_geotiffs_ds.drop_vars(2)
veg_geotiffs_ds = veg_geotiffs_ds.drop_vars(2)
dev_geotiffs_ds = dev_geotiffs_ds.drop_vars(2)
im_geotiffs_ds = im_geotiffs_ds.drop_vars(4)

print(water_geotiffs_ds.to_array().shape)
print(veg_geotiffs_ds.to_array().shape)
print(dev_geotiffs_ds.to_array().shape)
print(im_geotiffs_ds.to_array().shape)

#####################################################
#### make time-averages for filtering
#### water, veg, dev (no clipping)
for counter,g in tqdm(enumerate(geometries)):

    try:
        os.mkdir(f"../results/MR/MR_orthos_orig/region{counter}")
        os.mkdir(f"../results/MR/MR_dev/region{counter}")
        os.mkdir(f"../results/MR/MR_veg/region{counter}")
        os.mkdir(f"../results/MR/MR_water/region{counter}")
    except:
        pass

    veg_c = veg_geotiffs_ds.veg.rio.clip([g], veg_geotiffs_ds.veg.rio.crs)
    water_c = water_geotiffs_ds.water.rio.clip([g], water_geotiffs_ds.water.rio.crs)
    dev_c = dev_geotiffs_ds.dev.rio.clip([g], dev_geotiffs_ds.dev.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], im_geotiffs_ds.rio.crs)

    tmp = im_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/MR/MR_orthos_orig/region{counter}/Elwha_MR_region_{counter}_im_time_var_prob.tif", dtype=dtype)
    del tmp

    tmp = im_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/MR/MR_orthos_orig/region{counter}/Elwha_MR_region_{counter}_im_time_mean_prob.tif", dtype=dtype)
    del tmp

    tmp = veg_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/MR/MR_veg/region{counter}/Elwha_MR_region_{counter}_veg_time_var_prob.tif", dtype=dtype)
    del tmp

    tmp = veg_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/MR/MR_veg/region{counter}/Elwha_MR_region_{counter}_veg_time_mean_prob.tif", dtype=dtype)
    del tmp

    tmp = water_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/MR/MR_water/region{counter}/Elwha_MR_region_{counter}_water_time_var_prob.tif", dtype=dtype)
    del tmp

    tmp = water_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/MR/MR_water/region{counter}/Elwha_MR_region_{counter}_water_time_mean_prob.tif", dtype=dtype)
    del tmp

    tmp = dev_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/MR/MR_dev/region{counter}/Elwha_MR_region_{counter}_dev_time_var_prob.tif", dtype=dtype)
    del tmp

    tmp = dev_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/MR/MR_dev/region{counter}/Elwha_MR_region_{counter}_dev_time_mean_prob.tif", dtype=dtype)
    del tmp

#############################################################
#### recombine (mosaic) and regrid
if run_bash:

    os.chdir(f"../results/MR/MR_orthos_orig")
    os.system("bash mosaic_timeaverage.sh")
    os.chdir(cwd)

    os.chdir(f"../results/MR/MR_dev")
    os.system("bash mosaic_timeaverage.sh")
    os.chdir(cwd)

    os.chdir(f"../results/MR/MR_water")
    os.system("bash mosaic_timeaverage.sh")
    os.chdir(cwd)

    os.chdir(f"../results/MR/MR_veg")
    os.system("bash mosaic_timeaverage.sh")
    os.chdir(cwd)

#############################################################
#############################################################
#############################################################

#############################################################
##### chunk and mask wood based on region

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
wood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

wood_geotiffs_ds = wood_geotiffs_ds.drop_vars(2)
print(wood_geotiffs_ds.to_array().shape)

#############################################################
veg_mask_ds = rioxarray.open_rasterio("../results/MR/MR_veg/Elwha_MR_veg_time_bin0.9_regrid.tif", chunks=chunksize, dtype='uint8')
veg_mask_ds = veg_mask_ds.to_dataset('band')

dev_mask_ds = rioxarray.open_rasterio("../results/MR/MR_dev/Elwha_MR_dev_time_bin0.25_regrid.tif", chunks=chunksize, dtype='uint8')
dev_mask_ds = dev_mask_ds.to_dataset('band')

water_mask_ds = rioxarray.open_rasterio("../results/MR/MR_water/Elwha_MR_water_time_bin0.25_regrid.tif", chunks=chunksize, dtype='uint8')
water_mask_ds = water_mask_ds.to_dataset('band')

## clean up
water_mask_ds = water_mask_ds.drop_vars(2)
veg_mask_ds = veg_mask_ds.drop_vars(2)
dev_mask_ds = dev_mask_ds.drop_vars(2)

print(water_mask_ds.dims)
print(veg_mask_ds.dims)
print(dev_mask_ds.dims)

### filter wood
wood_geotiffs_ds = wood_geotiffs_ds.where((veg_mask_ds[1] < 1))
wood_geotiffs_ds = wood_geotiffs_ds.where((water_mask_ds[1] < 1))
wood_geotiffs_ds = wood_geotiffs_ds.where((dev_mask_ds[1] < 1))

print(wood_geotiffs_ds.dims)

#############################################################
#### make time-averages for filtering
#### water, veg, dev (no clipping)
for counter,g in tqdm(enumerate(geometries)):

    try:
        os.mkdir(f"../results/MR/MR_wood/region{counter}")
    except:
        pass

    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)

    tmp = wood_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/MR/MR_wood/region{counter}/region_{counter}_wood_time_var_prob.tif", dtype=dtype)
    del tmp

    tmp = wood_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/MR/MR_wood/region{counter}/region_{counter}_wood_time_mean_prob.tif", dtype=dtype)
    del tmp

    for time in times:
        tmp = wood_c.sel(time=time)
        tmp.rio.to_raster(raster_path=f"../results/MR/MR_wood/region{counter}/MR_{time}_region_{counter}_wood_prob.tif", dtype=dtype)
        del tmp        

        tmp = wood_c.sel(time=time)
        tmp.rio.to_raster(raster_path=f"../results/MR/MR_wood/region{counter}/MR_{time}_region_{counter}_wood_prob.tif", dtype=dtype)
        del tmp   

#############################################################
#### recombine (mosaic) and regrid
if run_bash:

    os.chdir(f"../results/MR/MR_wood")
    os.system("bash mosaic_timeaverage.sh")

    os.system("bash mosaic_t0.sh")
    os.system("bash mosaic_t1.sh")
    os.system("bash mosaic_t2.sh")
    os.system("bash mosaic_t3.sh")
    os.system("bash mosaic_t4.sh")
    os.system("bash mosaic_t5.sh")
    os.system("bash mosaic_t6.sh")
    os.system("bash mosaic_t7.sh")
    os.system("bash mosaic_t8.sh")
    os.system("bash mosaic_t9.sh")
    os.system("bash mosaic_t10.sh")
    os.system("bash mosaic_t11.sh")
    os.system("bash mosaic_t12.sh")
    os.system("bash mosaic_t13.sh")

    os.chdir(cwd)

    os.chdir(f"../results/MR/MR_wood/wood_detect")
    os.system("bash clip_all.sh")
    os.system("bash clip_all2.sh")
    os.chdir(cwd)