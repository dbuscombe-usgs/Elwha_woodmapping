## Dan Buscombe, Marda Science
## Apr, 2023
##
## Does this:
## 1. reads the Elwha_LR_*bin0.25_regrid_cc files into an xarray
## 2. reads the Elwha_LR_im_time_mean_prob image into an xarray
## 3. define a colormap that gets more blue with time
## 4. make a series of plots for each region showing wood locations
## 5. wood locations are color-coded by time on the plots
## 6. make an animation of the plots

## Where are we in the sequence?
## 1. filter_wood_by_av_veg_water_dev.py
## 2. >>>> make_wood_movies.py
## 3. timeseries_analysis.py

import os, json
import rioxarray
import xarray as xr 
from glob import glob 
import matplotlib.pyplot as plt
from dask.distributed import Client
from tqdm import tqdm
import matplotlib.colors
import numpy as np

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
## start client
client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

# Create variable used for time axis
time_var = xr.Variable('time',times)

######### get regions and clipper
regions = sorted(glob('../raw_data/GIS/LR*ID*.geojson'))
regions = [r for r in regions if 'pts' not in r]
print("{} regions".format(len(regions)))

geometries = []
for r in regions:
    with open(r) as f:
        gj = json.load(f)
    features = gj['features'][0]

    geometries.append(features['geometry'])

#############################################################
#############################################################
####################make wood movies

# get filtered wood probs, clipped to margins
wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/Elwha_LR_*bin0.25_regrid_cc.tif'))
print(len(wood_files))

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
wood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

# get timeaverage image for consistent lighting
avim_ds = rioxarray.open_rasterio("../results/LR/LR_orthos_orig/Elwha_LR_im_time_mean_prob.tif", chunks=chunksize, dtype='uint8')
avim_ds = avim_ds.to_dataset('band')
print(avim_ds.dims)
print(wood_geotiffs_ds.dims)

#############################################################
cmap=plt.cm.get_cmap('Blues', len(times))
custom_palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

for counter,g in tqdm(enumerate(geometries)):
    print("Working on region {}".format(counter))

    im_c = avim_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)

    tmp_da = xr.concat([im_c[1],im_c[2],im_c[3]],dim=('x','x','x'))
    
    sum_da = np.zeros((wood_c.dims['x'],wood_c.dims['y']))
    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        if inner_counter==0:
            fig1, ax1 = plt.subplots()
            plt.imshow(tmp_da.transpose()/255.)

        wood_da = wood_c.wood.sel(time=time)

        sum_da += wood_da.transpose().to_numpy()
        ## keep overlaying contours with deeper and deeper color with time
        CS1 = ax1.contour(wood_da.transpose(), colors=custom_palette[inner_counter])#, alpha=0.5)
        # plt.axis('off')

        # fmt = {}
        # strs = [time]
        # for l, s in zip(CS1.levels, strs):
        #     fmt[l] = s

        # # Label every other level using strings
        # ax1.clabel(CS1, CS1.levels[1::2], inline=True, fmt=fmt, fontsize=10)

    plt.axis('off')
    plt.savefig(f"../results/LR/LR_wood/summary/wood_animation/Wood_01_frame_{counter}_alltime.png", dpi=300, bbox_inches='tight')
    plt.close()
    del wood_da

    fig1, ax1 = plt.subplots()
    plt.imshow(tmp_da.transpose()/255.)
    sum_da[sum_da==0] = np.nan
    plt.imshow(100*(sum_da/len(times)), cmap='bwr')
    plt.axis('off')
    cb=plt.colorbar()
    cb.set_label('Percent wood occupancy')
    plt.savefig(f"../results/LR/LR_wood/summary/wood_animation/Wood_01_frame_{counter}_occupancy_alltime.png", dpi=300, bbox_inches='tight')
    plt.close()
    del sum_da

    tmp = wood_c.wood.sum("time", skipna=True)
    tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/region{counter}/Elwha_LR_region_{counter}_wood_sum_time.tif", dtype=dtype)
    del tmp

    del tmp_da, im_c, wood_c


#############################################################
if run_bash:

    for i in range(len(geometries)):
        try:
            os.mkdir(f"../results/LR/LR_wood/summary/wood_animation/region{i}")
        except:
            pass
        os.system(f'mv ../results/LR/LR_wood/summary/wood_animation/Wood_01_frame_{i}*.png ../results/LR/LR_wood/wood_animation/region{i}')

        os.system(f'convert -delay 100 ../results/LR/LR_wood/summary/wood_animation/region{i}/Wood_01_frame_{i}*.png ../results/LR/LR_wood/wood_animation/wood_animation_region{i}.gif')

    ### run bash script to stitch region sums
    os.chdir(f"../results/LR/LR_wood")
    os.system("bash mosaic_timesums.sh")
    os.chdir(cwd)        



#############################################################
#############################################################
####################make time-difference rasters

time0 = times[0]

for counter,g in tqdm(enumerate(geometries)):
    print("Working on region {}".format(counter))

    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
    
    for inner_counter, time in enumerate(times[1:]):
        print("Working on time {}".format(time))

        if inner_counter==0:
            wood_da0 = wood_c.wood.sel(time=time0)

        wood_da = wood_c.wood.sel(time=time)

        tmp = wood_da -  wood_da0

        tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/region{counter}/Elwha_LR_region_{counter}_wood_diff_time0.tif", dtype=dtype)
        del tmp

        del wood_da
    del wood_c



# #############################################################
# #############################################################
# ### whole-reach animation by time
# tmp_da = xr.concat([avim_ds[1],avim_ds[2],avim_ds[3]],dim=('x','x','x'))

# for inner_counter, time in enumerate(times):
#     print("Working on time {}".format(time))

#     if inner_counter==0:
#         plt.imshow(tmp_da.transpose()/255.)

#     wood_da = wood_geotiffs_ds.wood.sel(time=time)

#     plt.contour(wood_da.transpose(), colors=custom_palette[inner_counter], alpha=0.5)

# plt.axis('off')
# plt.savefig(f"../results/LR/LR_wood/summary/wood_animation/Wood_LR_frame_{counter}_allreach_alltime.png", dpi=300, bbox_inches='tight')
# plt.close()
# del wood_da
# del tmp_da

# #############################################################
# if run_bash:

#     for i in range(len(geometries)):
#         try:
#             os.mkdir(f"../results/LR/LR_wood/summary/region{i}")
#         except:
#             pass
#         os.system(f'mv ../results/LR/LR_wood/summary/Wood_01_frame_{i}*.png ../results/LR/LR_wood/summary/wood_animation/region{i}')

#         os.system(f'convert -delay 100 ../results/LR/LR_wood/summary/wood_animation/region{i}/Wood_01_frame_{i}*.png wood_animation_region{i}.gif')
