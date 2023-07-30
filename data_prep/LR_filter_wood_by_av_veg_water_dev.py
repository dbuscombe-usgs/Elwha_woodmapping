## Dan Buscombe, Marda Science
## Apr, 2023
##
## Does this:
## 1. calls bash scripts to use gdal to clip rasters to channel margins
## 2. then regrid to a common extent and pixel size
## 3. then make a datacube of Elwha LR wood, veg, water, dev
## 4. then create time-averages of veg, water, dev
## 5. then filter wood by time-averages of veg, water, dev
## 6. then chunk filtered wood by region

import json, os
import rioxarray
import xarray as xr 
from glob import glob 
from dask.distributed import Client
from tqdm import tqdm
from scipy import ndimage

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

run_bash = False #True

#############################################################

### recombine (mosaic) and regrid
if run_bash:
    os.chdir("../raw_data/LR/LR_orthos_orig/")
    os.system("bash regrid.sh")
    os.chdir(cwd)

    os.chdir("../raw_data/LR/LR_wood/")
    ## clip to channel margins
    os.system("../raw_data/bash regrid_cm.sh")
    # regrid to common extent and pixel size
    os.system("../raw_data/bash regrid.sh")
    os.chdir(cwd)

    os.chdir("../raw_data/LR/LR_dev/")
    os.system("bash regrid_cm.sh")
    os.system("bash regrid.sh")
    os.chdir(cwd)

    os.chdir("../raw_data/LR/LR_water/")
    os.system("bash regrid_cm.sh")
    os.system("bash regrid.sh")
    os.chdir(cwd)

    os.chdir("../raw_data/LR/LR_veg/")
    os.system("bash regrid_cm.sh")
    os.system("bash regrid.sh")
    os.chdir(cwd)

    os.chdir("../raw_data/LR/LR_all/")
    os.system("bash regrid_cm.sh")
    os.system("bash regrid.sh")
    os.system("bash extract_sed_grids.sh")
    os.system("bash extract_wood_grids.sh")
    os.chdir(cwd)

## read regridded "Prob" mosaic files
wood_files = sorted(glob('../raw_data/LR/LR_wood/LR_*_Prob1_regrid.tif'))
veg_files = sorted(glob('../raw_data/LR/LR_veg/LR_*_Prob1_regrid.tif'))
water_files = sorted(glob('../raw_data/LR/LR_water/LR_*_Prob0_regrid.tif'))
# dev_files = sorted(glob('../raw_data/LR/LR_dev/LR_*_Prob1_regrid.tif'))
im_files = sorted(glob('../raw_data/LR/LR_orthos_orig/Elwha_LR_*_regrid.tif'))
im_files = [i for i in im_files if 'bin' not in i]

sed_files = sorted(glob('../raw_data/LR/LR_all/LR_*sed*_regrid.tif'))
wood2_files = sorted(glob('../raw_data/LR/LR_all/LR_*wood2*_regrid.tif'))

print(len(wood2_files))
print(len(wood_files))
print(len(veg_files))
print(len(water_files))
# print(len(dev_files))
print(len(im_files))
print(len(times))
print(len(sed_files))

######### get regions 
regions = sorted(glob('../raw_data/GIS/LR*ID*_epsg6339.geojson'))
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

# #############################################################
# # Load in and concatenate all individual GeoTIFFs for devleopment
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dev_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# dev_geotiffs_ds = geotiffs_ds.rename({1: 'dev'})

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
# Load in and concatenate all individual GeoTIFFs for devleopment
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in sed_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
sed_geotiffs_ds = geotiffs_ds.rename({1: 'sed'})

#############################################################
## clean up
water_geotiffs_ds = water_geotiffs_ds.drop_vars(2)
veg_geotiffs_ds = veg_geotiffs_ds.drop_vars(2)
# dev_geotiffs_ds = dev_geotiffs_ds.drop_vars(2)
im_geotiffs_ds = im_geotiffs_ds.drop_vars(4)

print(water_geotiffs_ds.to_array().shape)
print(veg_geotiffs_ds.to_array().shape)
# print(dev_geotiffs_ds.to_array().shape)
print(im_geotiffs_ds.to_array().shape)
print(sed_geotiffs_ds.to_array().shape)

size = 8 ##1m
for time in times:
    print(time)
    tmp = water_geotiffs_ds.water.sel(time=time).to_numpy()
    tmp = ndimage.maximum_filter(tmp, size)
    water_geotiffs_ds.water.sel(time=time).data = tmp

#####################################################
#### make time-averages for filtering
#### water, veg, dev (no clipping)
for counter,g in tqdm(enumerate(geometries)):

    try:
        os.mkdir(f"../results/LR/LR_orthos_orig/region{counter}")
        # os.mkdir(f"../results/LR/LR_dev/region{counter}")
        os.mkdir(f"../results/LR/LR_veg/region{counter}")
        os.mkdir(f"../results/LR/LR_water/region{counter}")
        os.mkdir(f"../results/LR/LR_sed/region{counter}")
    except:
        pass

    veg_c = veg_geotiffs_ds.veg.rio.clip([g], veg_geotiffs_ds.veg.rio.crs)
    water_c = water_geotiffs_ds.water.rio.clip([g], water_geotiffs_ds.water.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], im_geotiffs_ds.rio.crs)
    sed_c = sed_geotiffs_ds.rio.clip([g], sed_geotiffs_ds.rio.crs)
    # dev_c = dev_geotiffs_ds.dev.rio.clip([g], dev_geotiffs_ds.dev.rio.crs)

    tmp = im_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_orthos_orig/region{counter}/Elwha_LR_region_{counter}_im_time_var_prob.tif", dtype=dtype)
    del tmp

    tmp = im_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_orthos_orig/region{counter}/Elwha_LR_region_{counter}_im_time_mean_prob.tif", dtype=dtype)
    del tmp

    tmp = veg_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_veg/region{counter}/Elwha_LR_region_{counter}_veg_time_var_prob.tif", dtype=dtype)
    del tmp

    tmp = veg_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_veg/region{counter}/Elwha_LR_region_{counter}_veg_time_mean_prob.tif", dtype=dtype)
    del tmp

    tmp = water_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_water/region{counter}/Elwha_LR_region_{counter}_water_time_var_prob.tif", dtype=dtype)
    del tmp

    tmp = water_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_water/region{counter}/Elwha_LR_region_{counter}_water_time_mean_prob.tif", dtype=dtype)
    del tmp

    # # tmp = dev_c.var("time", skipna=True)
    # # tmp.rio.to_raster(raster_path=f"../results/LR/LR_dev/region{counter}/Elwha_LR_region_{counter}_dev_time_var_prob.tif", dtype=dtype)
    # # del tmp

    # # tmp = dev_c.mean("time", skipna=True)
    # # tmp.rio.to_raster(raster_path=f"../results/LR/LR_dev/region{counter}/Elwha_LR_region_{counter}_dev_time_mean_prob.tif", dtype=dtype)
    # # del tmp

    tmp = sed_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_sed/region{counter}/Elwha_LR_region_{counter}_sed_time_mean_bin.tif", dtype=dtype)
    del tmp

    tmp = sed_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_sed/region{counter}/Elwha_LR_region_{counter}_sed_time_var_bin.tif", dtype=dtype)
    del tmp


#############################################################
#### recombine (mosaic) and regrid
if run_bash:

    os.chdir(f"../results/LR/LR_orthos_orig")
    os.system("bash mosaic_timeaverage.sh")
    os.chdir(cwd)

    # os.chdir(f"../results/LR/LR_dev")
    # os.system("bash mosaic_timeaverage.sh")
    # os.chdir(cwd)

    os.chdir(f"../results/LR/LR_water")
    os.system("bash mosaic_timeaverage.sh")
    os.chdir(cwd)

    os.chdir(f"../results/LR/LR_veg")
    os.system("bash mosaic_timeaverage.sh")
    os.chdir(cwd)

    os.chdir(f"../results/LR/LR_sed")
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

# size = 8 # 1m
# for time in times:
#     print(time)
#     tmp = wood_geotiffs_ds.wood.sel(time=time).to_numpy()
#     tmp = ndimage.maximum_filter(tmp, size)
#     wood_geotiffs_ds.wood.sel(time=time).data = tmp



# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood2_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
wood2_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

print(wood2_geotiffs_ds.to_array().shape)



#############################################################
veg_mask_ds = rioxarray.open_rasterio("../results/LR/LR_veg/Elwha_LR_veg_time_bin0.9_regrid.tif", chunks=chunksize, dtype='uint8')
veg_mask_ds = veg_mask_ds.to_dataset('band')

# dev_mask_ds = rioxarray.open_rasterio("../results/LR/LR_dev/Elwha_LR_dev_time_bin0.25_regrid.tif", chunks=chunksize, dtype='uint8')
# dev_mask_ds = dev_mask_ds.to_dataset('band')

water_mask_ds = rioxarray.open_rasterio("../results/LR/LR_water/Elwha_LR_water_time_bin0.5_regrid.tif", chunks=chunksize, dtype='uint8')
water_mask_ds = water_mask_ds.to_dataset('band')

# sed_mask_ds = rioxarray.open_rasterio("../results/LR/LR_sed/Elwha_LR_sed_time_bin0.9_regrid.tif", chunks=chunksize, dtype='uint8')
# sed_mask_ds = sed_mask_ds.to_dataset('band')

# dist_files = sorted(glob("../results/LR/LR_dist2braid/*.tif"))
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dist_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# dist_geotiffs_ds = geotiffs_ds.rename({1: 'dist'})

## clean up
water_mask_ds = water_mask_ds.drop_vars(2)
veg_mask_ds = veg_mask_ds.drop_vars(2)
# dev_mask_ds = dev_mask_ds.drop_vars(2)
# sed_mask_ds = sed_mask_ds.drop_vars(2)

print(water_mask_ds.dims)
print(veg_mask_ds.dims)
# print(dev_mask_ds.dims)
# print(sed_mask_ds.dims)

### filter wood

# print(wood_geotiffs_ds.sum().compute()) ##2.978e+07

wood_geotiffs_ds = wood_geotiffs_ds.where((veg_mask_ds[1] < 1))
wood_geotiffs_ds = wood_geotiffs_ds.where((water_mask_ds[1] < 1))
# wood_geotiffs_ds = wood_geotiffs_ds.where((dev_mask_ds[1] < 1))
# wood_geotiffs_ds = wood_geotiffs_ds.where((sed_mask_ds[1] < 1))
wood_geotiffs_ds = wood_geotiffs_ds.where((wood_geotiffs_ds.wood > 0))


wood2_geotiffs_ds = wood2_geotiffs_ds.where((veg_mask_ds[1] < 1))
wood2_geotiffs_ds = wood2_geotiffs_ds.where((water_mask_ds[1] < 1))
# wood2_geotiffs_ds = wood2_geotiffs_ds.where((dev_mask_ds[1] < 1))
# wood2_geotiffs_ds = wood2_geotiffs_ds.where((sed_mask_ds[1] < 1))
wood2_geotiffs_ds = wood2_geotiffs_ds.where((wood2_geotiffs_ds.wood > 0))

# print(wood_geotiffs_ds.sum().compute()) ##2.38e+07

#############################################################
#### make time-averages 
for counter,g in tqdm(enumerate(geometries)):

    try:
        os.mkdir(f"../results/LR/LR_wood/summary/region{counter}")
    except:
        pass

    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)

    tmp = wood_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/region{counter}/region_{counter}_wood_time_var_prob.tif", dtype=dtype)
    del tmp

    tmp = wood_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/region{counter}/region_{counter}_wood_time_mean_prob.tif", dtype=dtype)
    del tmp

    for time in times:
        tmp = wood_c.sel(time=time)
        tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/region{counter}/LR_{time}_region_{counter}_wood_prob.tif", dtype=dtype)
        del tmp        

        tmp = wood_c.sel(time=time)
        tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/region{counter}/LR_{time}_region_{counter}_wood_prob.tif", dtype=dtype)
        del tmp   
    del wood_c



#### make time-averages ,. "wood2"
for counter,g in tqdm(enumerate(geometries)):

    wood_c = wood2_geotiffs_ds.rio.clip([g], wood2_geotiffs_ds.rio.crs)

    tmp = wood_c.var("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/region{counter}/region_{counter}_wood2_time_var_prob.tif", dtype=dtype)
    del tmp

    tmp = wood_c.mean("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/region{counter}/region_{counter}_wood2_time_mean_prob.tif", dtype=dtype)
    del tmp

    for time in times:
        tmp = wood_c.sel(time=time)
        tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/region{counter}/LR_{time}_region_{counter}_wood2_prob.tif", dtype=dtype)
        del tmp        

        tmp = wood_c.sel(time=time)
        tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/region{counter}/LR_{time}_region_{counter}_wood2_prob.tif", dtype=dtype)
        del tmp   
    del wood_c




#############################################################
#### recombine (mosaic) and regrid
if run_bash:

    os.chdir(f"../results/LR/LR_wood/summary")
    os.system("bash mosaic_timeaverage.sh")
    os.system("bash wood2_mosaic_timeaverage.sh")

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

    os.system("bash wood2_mosaic_t0.sh")
    os.system("bash wood2_mosaic_t1.sh")
    os.system("bash wood2_mosaic_t2.sh")
    os.system("bash wood2_mosaic_t3.sh")
    os.system("bash wood2_mosaic_t4.sh")
    os.system("bash wood2_mosaic_t5.sh")
    os.system("bash wood2_mosaic_t6.sh")
    os.system("bash wood2_mosaic_t7.sh")
    os.system("bash wood2_mosaic_t8.sh")
    os.system("bash wood2_mosaic_t9.sh")
    os.system("bash wood2_mosaic_t10.sh")
    os.system("bash wood2_mosaic_t11.sh")
    os.system("bash wood2_mosaic_t12.sh")
    os.system("bash wood2_mosaic_t13.sh")


    os.system("mv *filtered_prob.tif ../wood_detect/")
    # os.system("mv *filtered_bin0.1_regrid.tif ../wood_detect/")
    os.system("mv *filtered_bin0.15_regrid.tif ../wood_detect/")

    os.chdir(cwd)

    os.chdir(f"../results/LR/LR_wood/wood_detect")

    os.system("bash clip_all.sh")
    os.system("bash clip_all2.sh")

    os.system("bash clip_all_wood2.sh")
    os.system("bash clip_all2_wood2.sh")

    os.system("bash add_wood1_wood2.sh")
    os.system("bash final_clip.sh")

    os.chdir(cwd)

# #### filter based on distance to braid

# if run_bash:
#     os.chdir(f"../results/LR/LR_wood/wood_detect")
#     os.system("bash filter_wood_by_dist.sh")
#     os.chdir(cwd)


# wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/Elwha_LR_*bin0.1_regrid_ccc.tif'))
# len(wood_files)