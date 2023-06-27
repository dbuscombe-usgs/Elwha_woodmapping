## Dan Buscombe, Marda Science
## Apr-June, 2023

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

# n_workers = 22
# threads_per_worker = 2
# memory_limit='115GB'
# ## start client
# client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

cwd = os.getcwd()

run_bash = False

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


r = "../raw_data/GIS/MR_movie_bars.geojson"
with open(r) as f:
    gj = json.load(f)
MR_bars = [a['geometry'] for a in gj['features']]


#############################################################
#########################################################
### regrid DEM rasters
### recombine (mosaic) and regrid
# all "results" rasters are 15928 x 41411
# pixel = 1.569605128802169152e-06 degrees (approx 15cm)
# gridded to extents of grid.geojson

if run_bash:
    os.chdir("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/")
    os.system("bash regridMR.sh") 
    os.chdir(cwd)

#############################################################
#########################################################
## time-series at every point
veg_files = sorted(glob('../raw_data/MR/MR_veg/MR_*_Prob1_regrid.tif'))
# water_files = sorted(glob('../raw_data/MR/MR_water/MR_*_Prob0_regrid.tif'))
# dev_files = sorted(glob('../raw_data/MR/MR_dev/MR_*_Prob1_regrid.tif'))
dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/*MR_*DEM_regrid.tif'))
print(len(dem_files))

sed_files = sorted(glob('../results/MR/MR_sed/Elwha_*sed.tif'))
print(len(sed_files))

# wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/Elwha_MR_*wood_filtered_bin0.1_regrid_final.tif'))
wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/model1/MR_*cleaned.tif'))

print(len(wood_files))

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
wood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

# #############################################################
# # Load in and concatenate all individual GeoTIFFs
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in water_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# water_geotiffs_ds = geotiffs_ds.rename({1: 'water'})

#############################################################
# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in veg_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
veg_geotiffs_ds = geotiffs_ds.rename({1: 'veg'})

#############################################################

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
dem_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})

#############################################################
# Load in and concatenate all individual GeoTIFFs for devleopment
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in sed_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
sed_geotiffs_ds = geotiffs_ds.rename({1: 'sed'})

#########################################################
## clean up
# water_geotiffs_ds = water_geotiffs_ds.drop_vars(2)
veg_geotiffs_ds = veg_geotiffs_ds.drop_vars(2)
# wood_geotiffs_ds = wood_geotiffs_ds.drop_vars(2)
dem_geotiffs_ds = dem_geotiffs_ds.drop_vars(2)

# print(water_geotiffs_ds.to_array().shape)
print(veg_geotiffs_ds.to_array().shape)
print(wood_geotiffs_ds.to_array().shape)
print(dem_geotiffs_ds.to_array().shape)
print(sed_geotiffs_ds.to_array().shape)

# get timeaverage image for consistent lighting
avim_ds = rioxarray.open_rasterio("../results/MR/MR_orthos_orig/Elwha_MR_im_time_mean_prob_regrid.tif", chunks=chunksize, dtype='uint8')
avim_ds = avim_ds.to_dataset('band')
print(avim_ds.dims)
print(wood_geotiffs_ds.dims)


# ####### MR
# MR_detrend_dem = rioxarray.open_rasterio("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_global_regrid.tif", chunks=chunksize, dtype=dtype)
# MR_detrend_dem = MR_detrend_dem.to_dataset('band').persist()
# print(MR_detrend_dem.dims)

# dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_2*.tif'))
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# dem_detrend_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})


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

#############################################################
#############################################################

#############################################################
#### focused movies


### wood chronology - different color per time
cmap = plt.get_cmap('inferno', len(times))
custom_palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
bounds=[0,1]

## wood 
for counter,g in tqdm(enumerate(MR_bars)):
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
        wood_da = ndimage.maximum_filter(wood_da, size=10)

        if inner_counter==0:
            fig1, ax1 = plt.subplots()
            ax1.imshow(matched)
        # ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
        wood_da[wood_da==0] = np.nan
        # ax1.imshow(wood_da, 'Reds_r', alpha=0.25)

        # make a color map of fixed colors
        cmap_tmp = matplotlib.colors.ListedColormap(['white', custom_palette[inner_counter]])
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        ax1.imshow(wood_da, interpolation='nearest', cmap=cmap_tmp, norm=norm, alpha=0.5)
        plt.axis('off')
        # plt.title(time)

        # plt.show()
        plt.savefig(f"../results/MR/MR_Bars_Wood_chron_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        # plt.close()
        del wood_da

    plt.close()





## wood 
for counter,g in tqdm(enumerate(MR_bars)):
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
        wood_da = ndimage.maximum_filter(wood_da, size=10)

        fig1, ax1 = plt.subplots()
        ax1.imshow(matched)
        ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
        wood_da[wood_da==0] = np.nan
        ax1.imshow(wood_da, 'Reds_r', alpha=0.25)
        plt.title(time)
        plt.axis('off')

        # plt.show()
        plt.savefig(f"../results/MR/MR_Bars_Wood_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del wood_da




## wood + sediment
for counter,g in tqdm(enumerate(MR_bars)):
    print("Working on region {}".format(counter))


    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
    sed_c = sed_geotiffs_ds.rio.clip([g], sed_geotiffs_ds.rio.crs)


    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.
        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
        wood_da = wood_c.wood.sel(time=time)
        wood_da = wood_da.transpose().to_numpy()
        wood_da = ndimage.maximum_filter(wood_da, size=10)

        sed_da = sed_c.sed.sel(time=time)
        sed_da = sed_da.transpose().to_numpy()
        sed_da[sed_da==0] = np.nan

        fig1, ax1 = plt.subplots()
        ax1.imshow(matched)
        ax1.imshow(sed_da, alpha=0.5, cmap='YlOrRd')

        ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
        wood_da[wood_da==0] = np.nan
        ax1.imshow(wood_da, 'Reds_r', alpha=0.25)
        plt.title(time)
        plt.axis('off')

        # plt.show()
        plt.savefig(f"../results/MR/MR_Bars_Wood_Sed_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del wood_da
        del sed_da



## wood age (sum)
for counter,g in tqdm(enumerate(MR_bars)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    sum_array = []
    for inner_counter, time in enumerate(times):
        # print("Working on time {}".format(time))
        wood_da = wood_c.wood.sel(time=time)
        wood_da = wood_da.transpose().to_numpy()
        wood_da = ndimage.maximum_filter(wood_da, size=10)
        sum_array.append(wood_da)

    sum_ = np.cumsum(np.dstack(sum_array),axis=-1)
    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))
        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.
        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

        fig1, ax1 = plt.subplots()
        ax1.imshow(matched)

        ax1.imshow(sum_[:,:,inner_counter], 'inferno', alpha=0.5)
        plt.title(time)
        plt.axis('off')

        # plt.show()
        plt.savefig(f"../results/MR/MR_Bars_Wood_age_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()



## sediment age (sum)
for counter,g in tqdm(enumerate(MR_bars)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    sed_c = sed_geotiffs_ds.rio.clip([g], sed_geotiffs_ds.rio.crs)
    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    sum_array = []
    for inner_counter, time in enumerate(times):
        # print("Working on time {}".format(time))
        sed_da = sed_c.sed.sel(time=time)
        sed_da = sed_da.transpose().to_numpy()
        sum_array.append(sed_da)

    sum_ = np.cumsum(np.dstack(sum_array),axis=-1)
    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))
        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.
        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

        fig1, ax1 = plt.subplots()
        ax1.imshow(matched)

        ax1.imshow(sum_[:,:,inner_counter], 'inferno', alpha=0.5)
        plt.title(time)
        plt.axis('off')

        # plt.show()
        plt.savefig(f"../results/MR/MR_Bars_Sed_age_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()


# #############################################################
# #### focused movies with HEIGHT contours

# # This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
# # then adds a percent sign.
# def fmt(x):
#     s = f"{x:.1f}"
#     if s.endswith("0"):
#         s = f"{x:.0f}"
#     return s


# ## wood 
# for counter,g in tqdm(enumerate(MR_bars)):
#     print("Working on region {}".format(counter))


#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
#     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
#     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))
#     dem_c = dem_detrend_geotiffs_ds.rio.clip([g], dem_detrend_geotiffs_ds.rio.crs)

#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))

#         im_da = tmp_da.sel(time=time).transpose()/255.
#         refim_da = reftmp_da.transpose()/255.
#         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         wood_da = ndimage.maximum_filter(wood_da, size=10)
#         dem_da = dem_c.dem.sel(time=time)
#         dem_da = dem_da.transpose().to_numpy()

#         fig1, ax1 = plt.subplots()
#         ax1.imshow(matched)
#         ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
#         wood_da[wood_da==0] = np.nan
#         ax1.imshow(wood_da, 'Reds_r', alpha=0.25)
#         plt.title(time)
#         plt.axis('off')

#         ## add DEM contours
#         CS = ax1.contour(dem_da)
#         ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=5)

#         # plt.show()
#         plt.savefig(f"../results/MR/MR_Bars_Wood_Contours_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da








# ## movie_geometries
# ## wood-sum over static image
# for counter,g in tqdm(enumerate(movie_geometries)):
#     print("Working on region {}".format(counter))

#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)

#     im_c = avim_ds.rio.clip([g], avim_ds.rio.crs)
#     tmp_da = xr.concat([im_c[1],im_c[2],im_c[3]],dim=('x','x','x'))
#     del im_c

#     fig1, ax1 = plt.subplots()
#     ax1.imshow(tmp_da.transpose()/255.)
#     del tmp_da
    
#     wood_da = wood_c.wood.sum("time", skipna=True).to_numpy()
#     wood_da[wood_da==0] = np.nan
#     im=ax1.imshow(wood_da.transpose()/len(times),cmap='Reds_r', vmin=0, vmax=1)
#     plt.colorbar(im, shrink=0.5)
#     del wood_da, wood_c
#     # plt.show()

#     plt.axis('off')
#     plt.savefig(f"../results/MR/Wood_woodsum_inst_movie_{counter}.png", dpi=300, bbox_inches='tight')
#     plt.close()


##### all
for counter,g in tqdm(enumerate(MR_bars)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
    veg_c = veg_geotiffs_ds.rio.clip([g], veg_geotiffs_ds.rio.crs)
    # water_c = water_geotiffs_ds.rio.clip([g], water_geotiffs_ds.rio.crs)
    dem_c = dem_geotiffs_ds.rio.clip([g], dem_geotiffs_ds.rio.crs)
    sed_c = sed_geotiffs_ds.rio.clip([g], sed_geotiffs_ds.rio.crs)

    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.

        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

        veg_da = veg_c.veg.sel(time=time)
        veg_da = veg_da.transpose().to_numpy()
        mask = veg_da==0

        veg_da = ndimage.maximum_filter(veg_da, size=10)
        veg_da[veg_da<.5] = np.nan  
        veg_da[veg_da>0]=1

        sed_da = sed_c.sed.sel(time=time)
        sed_da = sed_da.transpose().to_numpy().astype('float64')
        sed_da[sed_da<.5] = np.nan       

        wood_da = wood_c.wood.sel(time=time)
        wood_da = wood_da.transpose().to_numpy()
        # wood_da = ndimage.maximum_filter(wood_da, size=10)
        wood_da[wood_da==0] = np.nan

        water_da = np.ones_like(wood_da, dtype=np.float32)

        water_da[wood_da==1] = np.nan
        water_da[sed_da==1] = np.nan
        water_da[veg_da==1] = np.nan

        water_da[mask==1] = np.nan
        water_da[matched[:,:,0]==0] = np.nan


        # water_da = water_c.water.sel(time=time)
        # water_da = water_da.transpose().to_numpy()
        # water_da = ndimage.maximum_filter(water_da, size=10)
        # water_da[water_da<.2] = np.nan

        fig1, ax1 = plt.subplots()

        ax1.imshow(matched)
        ax1.imshow(water_da,'Blues_r', alpha=0.5)
        ax1.imshow(veg_da,'Purples_r', alpha=0.5)
        ax1.imshow(sed_da,'autumn_r', alpha=0.5)
        ax1.imshow(wood_da,'Reds_r')

        # dem_da = dem_c.dem.sel(time=time)
        # CS1 = ax1.contour(dem_da.transpose(), levels=8, cmap='Greys', alpha=0.5)
        # ax1.clabel(CS1, CS1.levels[1::2], inline=True, fontsize=5)
        plt.title(time)

        # plt.show()

        plt.axis('off')
        plt.savefig(f"../results/MR/All_inst_movie_bars_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del wood_da, water_da, veg_da, matched #dem_da, 




## wood only
for counter,g in tqdm(enumerate(movie_geometries)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        fig1, ax1 = plt.subplots()

        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.

        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

        ax1.imshow(matched)

        wood_da = wood_c.wood.sel(time=time)
        wood_da = wood_da.transpose().to_numpy()
        wood_da = ndimage.maximum_filter(wood_da, size=10)
        wood_da[wood_da==0] = np.nan
        ax1.imshow(wood_da,'Reds_r')
        plt.title(time)
        # plt.show()

        plt.axis('off')
        plt.savefig(f"../results/MR/Wood_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del wood_da



# ##### all regions

# ## wood only
# for counter,g in tqdm(enumerate(geometries)):
#     print("Working on region {}".format(counter))

#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
#     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
#     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))

#         fig1, ax1 = plt.subplots()

#         im_da = tmp_da.sel(time=time).transpose()/255.
#         refim_da = reftmp_da.transpose()/255.

#         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

#         ax1.imshow(matched)

#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         wood_da = ndimage.maximum_filter(wood_da, size=10)
#         wood_da[wood_da==0] = np.nan
#         ax1.imshow(wood_da,'Reds_r')
#         plt.title(time)
#         # plt.show()

#         plt.axis('off')
#         plt.savefig(f"../results/MR/Wood_inst_01_frame_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da


# ##### all
# for counter,g in tqdm(enumerate(geometries)):
#     print("Working on region {}".format(counter))

#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
#     veg_c = veg_geotiffs_ds.rio.clip([g], veg_geotiffs_ds.rio.crs)
#     water_c = water_geotiffs_ds.rio.clip([g], water_geotiffs_ds.rio.crs)
#     dem_c = dem_geotiffs_ds.rio.clip([g], dem_geotiffs_ds.rio.crs)
#     sed_c = sed_geotiffs_ds.rio.clip([g], sed_geotiffs_ds.rio.crs)

#     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
#     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))

#         fig1, ax1 = plt.subplots()

#         im_da = tmp_da.sel(time=time).transpose()/255.
#         refim_da = reftmp_da.transpose()/255.

#         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

#         ax1.imshow(matched)

#         water_da = water_c.water.sel(time=time)
#         water_da = water_da.transpose().to_numpy()
#         water_da = ndimage.maximum_filter(water_da, size=10)
#         water_da[water_da<.2] = np.nan
#         ax1.imshow(water_da,'Blues', alpha=0.5)

#         veg_da = veg_c.veg.sel(time=time)
#         veg_da = veg_da.transpose().to_numpy()
#         veg_da = ndimage.maximum_filter(veg_da, size=10)
#         veg_da[veg_da<.5] = np.nan        
#         ax1.imshow(veg_da,'Purples', alpha=0.5)

#         sed_da = sed_c.sed.sel(time=time)
#         sed_da = sed_da.transpose().to_numpy().astype('float64')
#         sed_da[sed_da<.5] = np.nan        
#         ax1.imshow(sed_da,'autumn_r', alpha=0.5)

#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         wood_da = ndimage.maximum_filter(wood_da, size=10)
#         wood_da[wood_da==0] = np.nan
#         ax1.imshow(wood_da,'Reds_r')

#         dem_da = dem_c.dem.sel(time=time)

#         CS1 = ax1.contour(dem_da.transpose(), levels=8, cmap='Greys', alpha=0.5)
#         ax1.clabel(CS1, CS1.levels[1::2], inline=True, fontsize=5)
#         plt.title(time)

#         # plt.show()

#         plt.axis('off')
#         plt.savefig(f"../results/MR/All_inst_01_frame_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da, dem_da, water_da, veg_da




# # #########################################
# # ################ movies with time-averaged imagery


# #############################################################
# # cmap=plt.cm.get_cmap('YlOrBr', len(times))
# # custom_palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

# for counter,g in tqdm(enumerate(geometries)):
#     print("Working on region {}".format(counter))

#     im_c = avim_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
#     veg_c = veg_geotiffs_ds.rio.clip([g], veg_geotiffs_ds.rio.crs)
#     water_c = water_geotiffs_ds.rio.clip([g], water_geotiffs_ds.rio.crs)
#     dem_c = dem_geotiffs_ds.rio.clip([g], dem_geotiffs_ds.rio.crs)

#     tmp_da = xr.concat([im_c[1],im_c[2],im_c[3]],dim=('x','x','x'))

#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))

#         fig1, ax1 = plt.subplots()
#         ax1.imshow(tmp_da.transpose()/255.)

#         water_da = water_c.water.sel(time=time)
#         water_da = water_da.transpose().to_numpy()
#         water_da[water_da<.2] = np.nan
#         ax1.imshow(water_da,'Blues', alpha=0.5)

#         veg_da = veg_c.veg.sel(time=time)
#         veg_da = veg_da.transpose().to_numpy()
#         veg_da[veg_da<.5] = np.nan        
#         ax1.imshow(veg_da,'Greens', alpha=0.5)

#         sed_da = sed_c.sed.sel(time=time)
#         sed_da = sed_da.transpose().to_numpy().astype('float64')
#         sed_da[sed_da<.5] = np.nan        
#         ax1.imshow(sed_da,'autumn_r', alpha=0.5)

#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         wood_da[wood_da==0] = np.nan
#         ax1.imshow(wood_da,'Reds_r')

#         # sed_da = np.zeros((wood_c.dims['x'],wood_c.dims['y']))
#         # sed_da[np.isnan(wood_da) & np.isnan(water_da) & np.isnan(veg_da)] = 1
#         # sed_da[sed_da==0] = np.nan
#         # ax1.imshow(sed_da,'YlGn')

#         dem_da = dem_c.dem.sel(time=time)

#         CS1 = ax1.contour(dem_da.transpose(), levels=5, cmap='YlOrBr', alpha=0.5)
#         ax1.clabel(CS1, CS1.levels, inline=True, fontsize=5) #[1::2]
#         plt.title(time)

#         # plt.show()

#         plt.axis('off')
#         plt.savefig(f"../results/MR/All_01_frame_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da, dem_da, water_da, veg_da




# #########################################
# ################ bin by elevation

# # x=np.array(points)[:,0]
# # y=np.array(points)[:,1]

# # dat_wood = np.zeros((len(x),len(times)))
# # dat_water = np.zeros((len(x),len(times)))
# # dat_veg = np.zeros((len(x),len(times)))
# # dat_dem = np.zeros((len(x),len(times)))

# # dem_geotiffs_ds.sel(time=times[0]).min().compute()

# # x=x[:100]
# # y=y[:100]

# # for counter, (xx,yy) in tqdm(enumerate(zip(x,y))):
# #     # pwood = wood_geotiffs_ds.wood.sel(x=xx,y=yy, method="nearest")
# #     # pwater = water_geotiffs_ds.water.sel(x=xx,y=yy, method="nearest")
# #     # pveg = veg_geotiffs_ds.veg.sel(x=xx,y=yy, method="nearest")
# #     pdem = dem_geotiffs_ds.dem.sel(x=xx,y=yy, method="nearest")
# #     print(pdem.to_numpy())

# #     dat_wood[counter,:] = pwood
# #     dat_water[counter,:] = pwater
# #     dat_veg[counter,:] = pveg
# #     dat_dem[counter,:] = pdem

# # np.savez('../results/MR/MR_wood/summary/bin_wood_water_veg_dem_allpts_5m.npz', dat_veg=dat_veg, dat_water=dat_water, dat_wood=dat_wood, dat_dem=dat_dem, x=x, y=y)


# #########################################
# ################ distance to nearest braid

