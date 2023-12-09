## Dan Buscombe, Marda Science
## 2023

import json, os
import rioxarray
import xarray as xr 
from glob import glob 
from dask.distributed import Client
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import ListedColormap

from skimage.exposure import match_histograms
import pandas as pd 
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

# n_workers = 10
# threads_per_worker = 2
# memory_limit='50GB'
# # start client
# client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

cwd = os.getcwd()

run_bash = False


# Create variable used for time axis
time_var = xr.Variable('time',times)

#############################################################
#########################################################
### regrid DEM rasters
### recombine (mosaic) and regrid
if run_bash:
    os.chdir("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/")
    os.system("bash regridLR.sh") 
    os.chdir(cwd)

######### get movie regions and clipper
movie_regions = sorted(glob('../raw_data/GIS/LR*movie*epsg6339.geojson'))
print("{} movie_ regions".format(len(movie_regions)))

movie_geometries = []
for r in movie_regions:
    with open(r) as f:
        gj = json.load(f)
    features = gj['features'][0]

    movie_geometries.append(features['geometry'])


r = "../raw_data/GIS/LR_movie_bars.geojson"
with open(r) as f:
    gj = json.load(f)
LR_bars = [a['geometry'] for a in gj['features']]


fourclass_files = sorted(glob('../results/LR/LR_all/model_out/*_4classMosaic.tif'))


#############################################################
# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in fourclass_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
label_geotiffs_ds = geotiffs_da.to_dataset('band')

dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/*LR_*DEM_regrid.tif'))


geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
dem_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})
dem_geotiffs_ds = dem_geotiffs_ds.drop_vars(2)


# get timeaverage image for consistent lighting
avim_ds = rioxarray.open_rasterio("../results/LR/LR_orthos_orig/Elwha_LR_im_time_mean_prob_regrid.tif", chunks=chunksize, dtype='uint8')
avim_ds = avim_ds.to_dataset('band')
print(avim_ds.dims)

#############################################################
im_files = sorted(glob('../raw_data/LR/LR_orthos_orig/Elwha_LR_*_regrid.tif'))
im_files = [i for i in im_files if 'bin' not in i]
print(len(im_files))

im_files_filt = []
for t in times:
    t = t.replace('-','')
    tmp = [d for d in im_files if t in d]
    im_files_filt.append(tmp[0])

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in im_files_filt],
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

## budget reaches
brfile = '../results/LR/LR_wood/wood_detect/model1/LR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
LRbudget_reaches = gj['features']

LRbudget_reaches_redo = []
for b in LRbudget_reaches:
    LRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))

brfile = '../results/LR/LR_wood/wood_detect/model1/LR_budget_reaches_epsg4326.geojson'
with open(brfile) as f:
    gj = json.load(f)
LRbudget_reaches2 = gj['features']


#############################################################
####  movies

### wood chronology - different color per time
cmap = plt.get_cmap('inferno', len(times))
custom_palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
bounds=[0,1]

mask = (label_geotiffs_ds[1]==4).astype(np.float)


## wood - chronology
for counter,g in tqdm(enumerate(LR_bars)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = mask.rio.clip([g], mask.rio.crs)
    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.
        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
        wood_da = wood_c.sel(time=time)
        wood_da = wood_da.transpose().to_numpy()
        # wood_da = ndimage.maximum_filter(wood_da, size=10)
        wood_da[wood_da<=0] = np.nan

        if inner_counter==0:
            fig1, ax1 = plt.subplots()
            ax1.imshow(matched)
        # ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
        # ax1.imshow(wood_da, 'Reds_r', alpha=0.25)

        # make a color map of fixed colors
        cmap_tmp = matplotlib.colors.ListedColormap(['white', custom_palette[inner_counter]])
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        ax1.imshow(wood_da, interpolation='nearest', cmap=cmap_tmp, norm=norm, alpha=0.5)
        plt.axis('off')
        # plt.title(time)

        # plt.show()
        plt.savefig(f"../results/LR/LR_Bars_Wood_chron_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        # plt.close()
        del wood_da

    plt.close()


## wood - age
for counter,g in tqdm(enumerate(LR_bars)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = mask.rio.clip([g], mask.rio.crs)
    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    sum_array = []
    for inner_counter, time in enumerate(times):
        # print("Working on time {}".format(time))
        wood_da = wood_c.sel(time=time)
        wood_da = wood_da.transpose().to_numpy()
        # wood_da = ndimage.maximum_filter(wood_da, size=10)
        wood_da[wood_da<=0] = np.nan
        sum_array.append(wood_da)

    sum_ = np.nancumsum(np.dstack(sum_array),axis=-1)

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
        plt.savefig(f"../results/LR/LR_Bars_Wood_age_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()

## wood - instantanues
for counter,g in tqdm(enumerate(LR_bars)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = mask.rio.clip([g], mask.rio.crs)
    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.
        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
        wood_da = wood_c.sel(time=time)
        wood_da = wood_da.transpose().to_numpy()
        # wood_da = ndimage.maximum_filter(wood_da, size=10)
        wood_da[wood_da<=0] = np.nan

        fig1, ax1 = plt.subplots()
        ax1.imshow(matched)
        # ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
        ax1.imshow(wood_da, 'Reds_r', alpha=0.5)
        plt.title(time)
        plt.axis('off')

        # plt.show()
        plt.savefig(f"../results/LR/LR_Bars_Wood_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del wood_da



sed_mask = (label_geotiffs_ds[1]==3).astype(np.float)


## wood + sediment
for counter,g in tqdm(enumerate(LR_bars)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    wood_c = mask.rio.clip([g], mask.rio.crs)
    sed_c = sed_mask.rio.clip([g], sed_mask.rio.crs)

    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.
        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
        wood_da = wood_c.sel(time=time)
        wood_da = wood_da.transpose().to_numpy()
        wood_da[wood_da<=0] = np.nan

        sed_da = sed_c.sel(time=time)
        sed_da = sed_da.transpose().to_numpy().astype('float')
        sed_da[sed_da<=0] = np.nan

        fig1, ax1 = plt.subplots()
        ax1.imshow(matched)
        ax1.imshow(sed_da, alpha=0.75, cmap='YlOrRd')

        # ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
        # wood_da[wood_da==0] = np.nan
        ax1.imshow(wood_da, 'Reds_r', alpha=0.5)
        plt.title(time)
        plt.axis('off')

        # plt.show()
        plt.savefig(f"../results/LR/LR_Bars_Wood_Sed_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del wood_da
        del sed_da


## sediment age (sum)
for counter,g in tqdm(enumerate(LR_bars)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    sed_c = sed_mask.rio.clip([g], sed_mask.rio.crs)
    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    sum_array = []
    for inner_counter, time in enumerate(times):
        # print("Working on time {}".format(time))
        sed_da = sed_c.sel(time=time)
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
        plt.savefig(f"../results/LR/LR_Bars_Sed_age_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()


cmap = ListedColormap(["black","lawngreen", "blue", "gold", "brown"])

## all - instantanues
for counter,g in tqdm(enumerate(LR_bars)):
    print("Working on region {}".format(counter))

    ref_c = reference.rio.clip([g], avim_ds.rio.crs)
    im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
    label_c = label_geotiffs_ds.rio.clip([g], label_geotiffs_ds.rio.crs)
    tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
    reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

    for inner_counter, time in enumerate(times):
        print("Working on time {}".format(time))

        im_da = tmp_da.sel(time=time).transpose()/255.
        refim_da = reftmp_da.transpose()/255.
        matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
        label_da = label_c[1].sel(time=time)
        label_da = label_da.transpose().to_numpy()
        # label_da[label_da<=0] = np.nan

        fig1, ax1 = plt.subplots()
        ax1.imshow(matched)
        ax1.imshow(label_da, cmap=cmap, alpha=0.3)
        plt.title(time)
        plt.axis('off')

        # plt.show()
        plt.savefig(f"../results/LR/LR_Bars_All_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del label_da



###############################
########### analysis of transition


# ## veg, water, sed, wood
# PM = []; O2M = []; O3M = []
# ## all - instantanues
# for counter,g in tqdm(enumerate(LR_bars)):
#     print("Working on region {}".format(counter))
#     label_c = label_geotiffs_ds.rio.clip([g], label_geotiffs_ds.rio.crs)

#     A = []
#     for x, y in zip(label_c.x, label_c.y):
#         tmp = label_c[1].sel(x=x, y=y).values
#         a = mc.MarkovChain().from_data(tmp)
#         A.append(a)

#     ind = np.where(np.array([len(a.observed_matrix) for a in A])==4)[0]
#     # np.array(A[ind].n_order_matrix)
#     PM.append(np.dstack([np.array(A[i].observed_p_matrix) for i in ind]).mean(axis=-1))
#     O2M.append(np.dstack([np.array(A[i].n_order_matrix(order=2)) for i in ind]).mean(axis=-1))
#     O3M.append(np.dstack([np.array(A[i].n_order_matrix(order=3)) for i in ind]).mean(axis=-1))

# np.savez('summaries/LR_transition_matrices.npz', LR_PM = PM, LR_O2M = O2M, LR_O3M = O3M)



# ###############################
# ########### analysis of transition


# ## veg, water, sed, wood
# PM = []; O2M = []; O3M = []
# ## all - instantanues
# for counter,g in tqdm(enumerate(LRbudget_reaches_redo[42:])):
#     print("Working on region {}".format(counter))
#     label_c = label_geotiffs_ds.rio.clip([g], label_geotiffs_ds.rio.crs)

#     A = []
#     for x, y in zip(label_c.x, label_c.y):
#         tmp = label_c[1].sel(x=x, y=y).values
#         a = mc.MarkovChain().from_data(tmp)
#         A.append(a)

#     ind = np.where(np.array([len(a.observed_matrix) for a in A])==4)[0]
#     try:
#         # np.array(A[ind].n_order_matrix)
#         PM.append(np.dstack([np.array(A[i].observed_p_matrix) for i in ind]).mean(axis=-1))
#         O2M.append(np.dstack([np.array(A[i].n_order_matrix(order=2)) for i in ind]).mean(axis=-1))
#         O3M.append(np.dstack([np.array(A[i].n_order_matrix(order=3)) for i in ind]).mean(axis=-1))
#     except:
#         PM.append(np.ones((4,4))*np.nan)
#         O2M.append(np.ones((4,4))*np.nan)
#         O3M.append(np.ones((4,4))*np.nan)

# np.savez('summaries/LR_transition_matrices_budgetreaches_partial42_52.npz', LR_PM = PM, LR_O2M = O2M, LR_O3M = O3M)


# # np.savez('summaries/LR_transition_matrices_budgetreaches_partial0_41.npz', LR_PM = PM, LR_O2M = O2M, LR_O3M = O3M)

# # np.savez('summaries/LR_transition_matrices_budgetreaches.npz', LR_PM = PM, LR_O2M = O2M, LR_O3M = O3M)


##################################################################


# dists = pd.read_csv('br_dists.csv')
# LR = np.hstack((0,np.array(dists['LR'])))
# MR = np.hstack((0,np.array(dists['MR'][:43])))


# def rescale_array(dat, mn, mx):
#     """
#     rescales an input dat between mn and mx
#     Code from doodleverse_utils by Daniel Buscombe
#     source: https://github.com/Doodleverse/doodleverse_utils
#     """
#     m = min(dat.flatten())
#     M = max(dat.flatten())
#     return (mx - mn) * (dat - m) / (M - m) + mn


# LR = rescale_array(LR,11,2)
# MR = rescale_array(MR[::-1],12,20)

# ##################################################################


# with np.load('summaries/LR_transition_matrices_budgetreaches.npz', allow_pickle=True) as dat:
#     LR_tpm = dict()
#     for k in dat.keys():
#         LR_tpm[k] = dat[k]
#     del dat

# with np.load('summaries/MR_transition_matrices_budgetreaches.npz', allow_pickle=True) as dat:
#     MR_tpm = dict()
#     for k in dat.keys():
#         MR_tpm[k] = dat[k]
#     del dat

# MR_TPM = []
# for k in MR_tpm['MR_PM']:
#     if np.isnan(k).any():
#         tmp = np.ones((4,4))*np.nan
#         MR_TPM.append(tmp)
#     else:
#         MR_TPM.append(k)

# LR_TPM = []
# for k in LR_tpm['LR_PM']:
#     if np.isnan(k).any():
#         tmp = np.ones((4,4))*np.nan
#         LR_TPM.append(tmp)
#     else:
#         LR_TPM.append(k)


# LRxy = np.array([np.mean(np.mean(g['coordinates'],axis=0),axis=1).flatten() for g in LRbudget_reaches_redo])
# LRx = LRxy[:,0]
# LRy = LRxy[:,1]

# LR_wood_pers = [p[3,3] for p in LR_TPM]#
# LR_sed_pers = [p[2,2] for p in  LR_TPM]#MR_tpm['MR_PM']]
# LR_veg_pers = [p[0,0] for p in  LR_TPM]#MR_tpm['MR_PM']]
# LR_water_pers = [p[1,1] for p in  LR_TPM]#MR_tpm['MR_PM']]
# LR_veg_encroach = [p[0,1] for p in  LR_TPM]#MR_tpm['MR_PM']]
# LR_veg_growth = [p[0,2] for p in LR_TPM]# MR_tpm['MR_PM']]
# LR_wood_occl = [p[0,3] for p in  LR_TPM]#MR_tpm['MR_PM']]
# LR_veg_erosion = [p[1,0] for p in  LR_TPM]#MR_tpm['MR_PM']]
# LR_sed_erosion = [p[1,2] for p in  LR_TPM]#MR_tpm['MR_PM']]
# LR_wood_erosion = [p[1,3] for p in  LR_TPM]#MR_tpm['MR_PM']]

# MR_wood_pers = [p[3,3] for p in MR_TPM]#
# MR_sed_pers = [p[2,2] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_veg_pers = [p[0,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_water_pers = [p[1,1] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_veg_encroach = [p[0,1] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_veg_growth = [p[0,2] for p in MR_TPM]# MR_tpm['MR_PM']]
# MR_wood_occl = [p[0,3] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_veg_erosion = [p[1,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_sed_erosion = [p[1,2] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_wood_erosion = [p[1,3] for p in  MR_TPM]#MR_tpm['MR_PM']]

# species = ([str(k) for k in range(len(MR_wood_pers))])

# MR_weight_counts = {
#     "Wood": np.array(MR_wood_pers),
#     "Sed": np.array(MR_sed_pers),
#     "Veg": np.array(MR_veg_pers),
#     "Water": np.array(MR_water_pers)
# }

# LR_weight_counts = {
#     "Wood": np.array(LR_wood_pers),
#     "Sed": np.array(LR_sed_pers),
#     "Veg": np.array(LR_veg_pers),
#     "Water": np.array(LR_water_pers)
# }
# width = 0.5

# fig, ax = plt.subplots()
# bottom = np.zeros(len(species))

# for boolean, weight_count in MR_weight_counts.items():
#     p = ax.bar(times[1:], weight_count, width, label=boolean, bottom=bottom)
#     bottom += weight_count

# ax.set_title("Landcover persistence")
# ax.legend(loc="upper right")

# plt.show()


# fig, ax = plt.subplots()
# bottom = np.zeros(len(species))

# for boolean, weight_count in LR_weight_counts.items():
#     p = ax.bar(times[1:], weight_count, width, label=boolean, bottom=bottom)
#     bottom += weight_count

# ax.set_title("Landcover persistence")
# ax.legend(loc="upper right")

# plt.show()

##################################################################


# # np.median(PM,axis=0)

# with np.load('summaries/LR_transition_matrices.npz', allow_pickle=True) as dat:
#     LR_tpm = dict()
#     for k in dat.keys():
#         LR_tpm[k] = dat[k]
#     del dat

# with np.load('summaries/MR_transition_matrices.npz', allow_pickle=True) as dat:
#     MR_tpm = dict()
#     for k in dat.keys():
#         MR_tpm[k] = dat[k]
#     del dat

# MR_TPM = []
# for k in MR_tpm['MR_PM']:
#     if np.isnan(k).any():
#         tmp = np.ones((4,4))*np.nan
#         MR_TPM.append(tmp)
#     else:
#         MR_TPM.append(k)

# LR_TPM = []
# for k in LR_tpm['LR_PM']:
#     if np.isnan(k).any():
#         tmp = np.ones((4,4))*np.nan
#         LR_TPM.append(tmp)
#     else:
#         LR_TPM.append(k)


# xy = np.array([np.mean(np.mean(g['coordinates'],axis=0),axis=1).flatten() for g in LR_bars])
# x = xy[:,0]
# y = xy[:,1]


# MR_wood_pers = [p[3,3] for p in MR_TPM]#
# MR_sed_pers = [p[2,2] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_veg_pers = [p[0,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_water_pers = [p[1,1] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_veg_encroach = [p[0,1] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_veg_growth = [p[0,2] for p in MR_TPM]# MR_tpm['MR_PM']]
# MR_wood_occl = [p[0,3] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_veg_erosion = [p[1,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_sed_erosion = [p[1,2] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_wood_erosion = [p[1,3] for p in  MR_TPM]#MR_tpm['MR_PM']]

# species = ([str(k) for k in range(len(MR_wood_pers))])

# weight_counts = {
#     "Wood": np.array(MR_wood_pers),
#     "Sed": np.array(MR_sed_pers),
#     "Veg": np.array(MR_veg_pers),
#     "Water": np.array(MR_water_pers)
# }
# width = 0.5

# fig, ax = plt.subplots()
# bottom = np.zeros(len(species))

# for boolean, weight_count in weight_counts.items():
#     p = ax.bar(times[1:], weight_count, width, label=boolean, bottom=bottom)
#     bottom += weight_count

# ax.set_title("Landcover persistence")
# ax.legend(loc="upper right")

# plt.show()






# # plt.plot(MR_wood_pers, MR_sed_pers,'ro')
# plt.plot(MR_wood_pers, MR_veg_pers,'gs')
# # plt.plot(MR_wood_pers, MR_water_pers,'bh')
# plt.show()

# plt.plot(MR_wood_pers, MR_sed_erosion,'ro')
# plt.plot(MR_wood_pers, MR_wood_erosion,'go')
# plt.plot(MR_wood_pers, MR_veg_erosion,'bo')
# plt.show()



# MR_tmp = np.nanmedian(MR_TPM,axis=0)
# LR_tmp = np.nanmedian(LR_TPM,axis=0)


# plt.figure(figsize=(12,8))
# plt.subplot(221)
# g = sns.heatmap(MR_tmp, annot = True, cmap ='plasma', vmax=1, vmin=0,
#             linecolor ='black', linewidths = 1, cbar_kws={'label': 'Probability of transition'})

# g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
# g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])

# plt.subplot(222)
# g = sns.heatmap(LR_tmp, annot = True, cmap ='plasma', vmax=1, vmin=0,
#             linecolor ='black', linewidths = 1, cbar_kws={'label': 'Probability of transition'})

# g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
# g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])


# # plt.show()

# plt.savefig("summaries/median_reach_TPM.png", dpi=300, bbox_inches="tight")
# plt.close()



# plt.figure(figsize=(12,8))
# plt.subplot(221)
# g = sns.heatmap(MR_tmp-LR_tmp, annot = True, cmap ='bwr', vmax=.2, vmin=-.2,
#             linecolor ='black', linewidths = 1, cbar_kws={'label': 'Probability of transition'})

# g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
# g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])

# # plt.show()

# plt.savefig("summaries/median_reach_TPM_diff.png", dpi=300, bbox_inches="tight")
# plt.close()



# plt.figure(figsize=(12,8))
# plt.subplot(221)
# g = sns.heatmap(np.nanstd(np.dstack(MR_TPM),axis=-1)/np.nanmean(np.dstack(MR_TPM),axis=-1), annot = True, cmap ='bwr', vmin=.3, vmax=1.2,
#             linecolor ='black', linewidths = 1, cbar_kws={'label': 'Spatial variability\n of transition (-)'})

# g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
# g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
# plt.title('a) MR', loc='left'); 

# plt.subplot(222)
# g = sns.heatmap(np.nanstd(np.dstack(LR_TPM),axis=-1)/np.nanmean(np.dstack(LR_TPM),axis=-1), annot = True, cmap ='bwr', vmin=.3, vmax=1.2,
#             linecolor ='black', linewidths = 1, cbar_kws={'label': 'Spatial variability\n of transition (-)'})

# g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
# g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
# plt.title('b) LR', loc='left'); 

# # plt.show()

# plt.savefig("summaries/TPM_spatial_var.png", dpi=300, bbox_inches="tight")
# plt.close()

# from skimage import measure
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
   
# dx, dy,dz = np.gradient(np.dstack(MR_TPM))
# iso_val=0.15
# # verts, faces, normals, values = measure.marching_cubes(np.dstack(MR_TPM), iso_val, spacing=(0.01, 0.01, 0.01))
# verts, faces, normals, values = measure.marching_cubes(dy, iso_val, spacing=(0.001, 0.001, 0.001))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
#                 cmap='Spectral', lw=1)
# plt.show()










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
#     plt.savefig(f"../results/LR/Wood_woodsum_inst_movie_{counter}.png", dpi=300, bbox_inches='tight')
#     plt.close()


# ## wood only
# for counter,g in tqdm(enumerate(movie_geometries)):
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
#         plt.savefig(f"../results/LR/Wood_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da



# ##### all
# for counter,g in tqdm(enumerate(LR_bars)):
#     print("Working on region {}".format(counter))

#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
#     veg_c = veg_geotiffs_ds.rio.clip([g], veg_geotiffs_ds.rio.crs)
#     # water_c = water_geotiffs_ds.rio.clip([g], water_geotiffs_ds.rio.crs)
#     dem_c = dem_geotiffs_ds.rio.clip([g], dem_geotiffs_ds.rio.crs)
#     sed_c = sed_geotiffs_ds.rio.clip([g], sed_geotiffs_ds.rio.crs)

#     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
#     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))

#         im_da = tmp_da.sel(time=time).transpose()/255.
#         refim_da = reftmp_da.transpose()/255.

#         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

#         veg_da = veg_c.veg.sel(time=time)
#         veg_da = veg_da.transpose().to_numpy()
#         mask = veg_da==0

#         veg_da = ndimage.maximum_filter(veg_da, size=10)
#         veg_da[veg_da<.5] = np.nan  
#         veg_da[veg_da>0]=1

#         sed_da = sed_c.sed.sel(time=time)
#         sed_da = sed_da.transpose().to_numpy().astype('float64')
#         sed_da[sed_da<.5] = np.nan       

#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         # wood_da = ndimage.maximum_filter(wood_da, size=10)
#         wood_da[wood_da==0] = np.nan

#         water_da = np.ones_like(wood_da, dtype=np.float32)

#         water_da[wood_da==1] = np.nan
#         water_da[sed_da==1] = np.nan
#         water_da[veg_da==1] = np.nan

#         water_da[mask==1] = np.nan
#         water_da[matched[:,:,0]==0] = np.nan


#         # water_da = water_c.water.sel(time=time)
#         # water_da = water_da.transpose().to_numpy()
#         # water_da = ndimage.maximum_filter(water_da, size=10)
#         # water_da[water_da<.2] = np.nan

#         fig1, ax1 = plt.subplots()

#         ax1.imshow(matched)
#         ax1.imshow(water_da,'Blues_r', alpha=0.5)
#         ax1.imshow(veg_da,'Purples_r', alpha=0.5)
#         ax1.imshow(sed_da,'autumn_r', alpha=0.5)
#         ax1.imshow(wood_da,'Reds_r')

#         # dem_da = dem_c.dem.sel(time=time)
#         # CS1 = ax1.contour(dem_da.transpose(), levels=8, cmap='Greys', alpha=0.5)
#         # ax1.clabel(CS1, CS1.levels[1::2], inline=True, fontsize=5)
#         plt.title(time)

#         # plt.show()

#         plt.axis('off')
#         plt.savefig(f"../results/LR/All_inst_movie_bars_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da, water_da, veg_da, matched #dem_da, 









# #############################################################
# #########################################################

# veg_files = sorted(glob('../raw_data/LR/LR_veg/LR_*_Prob1_regrid.tif'))
# # water_files = sorted(glob('../raw_data/LR/LR_water/LR_*_Prob0_regrid.tif'))
# dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/*LR_*DEM_regrid.tif'))
# print(len(dem_files))

# # sed_files = sorted(glob('../raw_data/LR/LR_all/LR_*sed*_regrid.tif'))
# sed_files = sorted(glob('../results/LR/LR_sed/Elwha_*sed.tif'))

# print(len(sed_files))

# wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/model1/LR_*cleaned.tif'))

# print(len(wood_files))

# # Load in and concatenate all individual GeoTIFFs
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# wood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

# # #############################################################
# # # Load in and concatenate all individual GeoTIFFs
# # geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in water_files],
# #                         dim=time_var)
# # # Covert our xarray.DataArray into a xarray.Dataset
# # geotiffs_ds = geotiffs_da.to_dataset('band')
# # # Rename the variable to a more useful name
# # water_geotiffs_ds = geotiffs_ds.rename({1: 'water'})

# #############################################################
# # Load in and concatenate all individual GeoTIFFs
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in veg_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# veg_geotiffs_ds = geotiffs_ds.rename({1: 'veg'})

# #############################################################

# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# dem_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})

# #############################################################
# # Load in and concatenate all individual GeoTIFFs for devleopment
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in sed_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# sed_geotiffs_ds = geotiffs_ds.rename({1: 'sed'})

# #########################################################
# ## clean up
# # water_geotiffs_ds = water_geotiffs_ds.drop_vars(2)
# veg_geotiffs_ds = veg_geotiffs_ds.drop_vars(2)
# # wood_geotiffs_ds = wood_geotiffs_ds.drop_vars(2)
# dem_geotiffs_ds = dem_geotiffs_ds.drop_vars(2)

# # print(water_geotiffs_ds.to_array().shape)
# print(veg_geotiffs_ds.to_array().shape)
# print(wood_geotiffs_ds.to_array().shape)
# print(dem_geotiffs_ds.to_array().shape)

# # get timeaverage image for consistent lighting
# avim_ds = rioxarray.open_rasterio("../results/LR/LR_orthos_orig/Elwha_LR_im_time_mean_prob_regrid.tif", chunks=chunksize, dtype='uint8')
# avim_ds = avim_ds.to_dataset('band')
# print(avim_ds.dims)
# print(wood_geotiffs_ds.dims)


# #########################################
# ################ movies with histogram-matched imagery




# #############################################################
# im_files = sorted(glob('../raw_data/LR/LR_orthos_orig/Elwha_LR_*_regrid.tif'))
# im_files = [i for i in im_files if 'bin' not in i]
# print(len(im_files))

# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in im_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# im_geotiffs_ds = geotiffs_ds.rename({1: 'red'})
# im_geotiffs_ds = im_geotiffs_ds.rename({2: 'green'})
# im_geotiffs_ds = im_geotiffs_ds.rename({3: 'blue'})
# im_geotiffs_ds = im_geotiffs_ds.drop_vars(4)
# print(im_geotiffs_ds.to_array().shape)
# print(sed_geotiffs_ds.to_array().shape)

# ### reference image (bright)
# reference = im_geotiffs_ds.sel(time='2016-07-14')



##########################################################################################


# ################################################################
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
#         plt.savefig(f"../results/LR/Wood_inst_01_frame_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da


##### alll

# #############################################################
# im_files = sorted(glob('../raw_data/LR/LR_orthos_orig/Elwha_LR_*_regrid.tif'))
# im_files = [i for i in im_files if 'bin' not in i]
# print(len(im_files))

# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in im_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# im_geotiffs_ds = geotiffs_ds.rename({1: 'red'})
# im_geotiffs_ds = im_geotiffs_ds.rename({2: 'green'})
# im_geotiffs_ds = im_geotiffs_ds.rename({3: 'blue'})
# im_geotiffs_ds = im_geotiffs_ds.drop_vars(4)
# print(im_geotiffs_ds.to_array().shape)

# ### reference image (bright)
# reference = im_geotiffs_ds.sel(time='2016-07-14')


# for counter,g in tqdm(enumerate(geometries)):
#     print("Working on region {}".format(counter))

#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
#     veg_c = veg_geotiffs_ds.rio.clip([g], veg_geotiffs_ds.rio.crs)
#     water_c = water_geotiffs_ds.rio.clip([g], water_geotiffs_ds.rio.crs)
#     dem_c = dem_geotiffs_ds.rio.clip([g], dem_geotiffs_ds.rio.crs)

#     # tmp_da = xr.concat([im_c[1],im_c[2],im_c[3]],dim=('x','x','x'))
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

#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         wood_da = ndimage.maximum_filter(wood_da, size=10)
#         wood_da[wood_da==0] = np.nan
#         ax1.imshow(wood_da,'Reds_r')

#         # sed_da = np.zeros((wood_c.dims['x'],wood_c.dims['y']))
#         # sed_da[np.isnan(wood_da) & np.isnan(water_da) & np.isnan(veg_da)] = 1
#         # sed_da[sed_da==0] = np.nan
#         # ax1.imshow(sed_da,'YlGn')

#         dem_da = dem_c.dem.sel(time=time)

#         CS1 = ax1.contour(dem_da.transpose(), levels=8, cmap='YlOrBr', alpha=0.5)
#         ax1.clabel(CS1, CS1.levels[1::2], inline=True, fontsize=5)
#         plt.title(time)

#         # plt.show()

#         plt.axis('off')
#         plt.savefig(f"../results/LR/All_inst_01_frame_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
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
#         water_da = ndimage.maximum_filter(water_da, size=10)
#         water_da[water_da<.2] = np.nan
#         ax1.imshow(water_da,'Blues', alpha=0.5)

#         veg_da = veg_c.veg.sel(time=time)
#         veg_da = veg_da.transpose().to_numpy()
#         veg_da = ndimage.maximum_filter(veg_da, size=10)
#         veg_da[veg_da<.5] = np.nan        
#         ax1.imshow(veg_da,'Purples', alpha=0.5)

#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         wood_da = ndimage.maximum_filter(wood_da, size=10)
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
#         plt.savefig(f"../results/LR/All_01_frame_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da, dem_da, water_da, veg_da



# #########################################
# ################ bin by elevation

# x=np.array(points)[:,0]
# y=np.array(points)[:,1]

# dat_wood = np.zeros((len(x),len(times)))
# dat_water = np.zeros((len(x),len(times)))
# dat_veg = np.zeros((len(x),len(times)))
# dat_dem = np.zeros((len(x),len(times)))

# dem_geotiffs_ds.sel(time=times[0]).min().compute()

# x=x[:100]
# y=y[:100]

# for counter, (xx,yy) in tqdm(enumerate(zip(x,y))):
#     # pwood = wood_geotiffs_ds.wood.sel(x=xx,y=yy, method="nearest")
#     # pwater = water_geotiffs_ds.water.sel(x=xx,y=yy, method="nearest")
#     # pveg = veg_geotiffs_ds.veg.sel(x=xx,y=yy, method="nearest")
#     pdem = dem_geotiffs_ds.dem.sel(x=xx,y=yy, method="nearest")
#     print(pdem.to_numpy())

#     dat_wood[counter,:] = pwood
#     dat_water[counter,:] = pwater
#     dat_veg[counter,:] = pveg
#     dat_dem[counter,:] = pdem

# np.savez('../results/LR/LR_wood/summary/bin_wood_water_veg_dem_allpts_5m.npz', dat_veg=dat_veg, dat_water=dat_water, dat_wood=dat_wood, dat_dem=dat_dem, x=x, y=y)


#########################################
################ distance to nearest braid

