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
from skimage.exposure import match_histograms
import pandas as pd 

from matplotlib.colors import ListedColormap

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

# Create variable used for time axis
time_var = xr.Variable('time',times)

#############################################################
#########################################################
### regrid DEM rasters
### recombine (mosaic) and regrid
if run_bash:
    os.chdir("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/")
    os.system("bash regridMR.sh") 
    os.chdir(cwd)

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

## budget reaches
brfile = '../results/MR/MR_wood/wood_detect/model1/MR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
MRbudget_reaches = gj['features']

MRbudget_reaches_redo = []
for b in MRbudget_reaches:
    MRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))

brfile = '../results/MR/MR_wood/wood_detect/model1/MR_budget_reaches_epsg4326.geojson'
with open(brfile) as f:
    gj = json.load(f)
MRbudget_reaches2 = gj['features']


#############################################################
dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/*MR_*DEM_regrid.tif'))

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
dem_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})
dem_geotiffs_ds = dem_geotiffs_ds.drop_vars(2)


# get timeaverage image for consistent lighting
avim_ds = rioxarray.open_rasterio("../results/MR/MR_orthos_orig/Elwha_MR_im_time_mean_prob_regrid.tif", chunks=chunksize, dtype='uint8')
avim_ds = avim_ds.to_dataset('band')
print(avim_ds.dims)

#############################################################
im_files = sorted(glob('../raw_data/MR/MR_orthos_orig/Elwha_MR_*_regrid.tif'))
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

#############################################################
fourclass_files = sorted(glob('../results/MR/MR_all/model_out/*_4classMosaic.tif'))

#############################################################
# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in fourclass_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
label_geotiffs_ds = geotiffs_da.to_dataset('band')


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


##################################################################



#############################################################
#### focused movies

### wood chronology - different color per time
cmap = plt.get_cmap('inferno', len(times))
custom_palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
bounds=[0,1]

mask = (label_geotiffs_ds[1]==4).astype(np.float)


## wood - chronology
for counter,g in tqdm(enumerate(MR_bars)):
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
        plt.savefig(f"../results/MR/MR_Bars_Wood_chron_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        # plt.close()
        del wood_da

    plt.close()




sed_mask = (label_geotiffs_ds[1]==3).astype(np.float)


## wood + sediment
for counter,g in tqdm(enumerate(MR_bars)):
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
        plt.savefig(f"../results/MR/MR_Bars_Wood_Sed_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del wood_da
        del sed_da




## sediment age (sum)
for counter,g in tqdm(enumerate(MR_bars)):
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
        plt.savefig(f"../results/MR/MR_Bars_Sed_age_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()


cmap = ListedColormap(["black","lawngreen", "blue", "gold", "brown"])

## all - instantanues
for counter,g in tqdm(enumerate(MR_bars)):
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
        plt.savefig(f"../results/MR/MR_Bars_All_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del label_da


## wood - age
for counter,g in tqdm(enumerate(MR_bars)):
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
        plt.savefig(f"../results/MR/MR_Bars_Wood_age_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()

## wood - instantanues
for counter,g in tqdm(enumerate(MR_bars)):
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
        plt.savefig(f"../results/MR/MR_Bars_Wood_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del wood_da




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


###############################
########### analysis of transition


# ## veg, water, sed, wood
# PM = []; O2M = []; O3M = []
# ## all - instantanues
# for counter,g in tqdm(enumerate(MR_bars[7:])):
#     print("Working on region {}".format(7+counter))
#     label_c = label_geotiffs_ds.rio.clip([g], label_geotiffs_ds.rio.crs)

#     A = []
#     for x, y in zip(label_c.x, label_c.y):
#         tmp = label_c[1].sel(x=x, y=y).values
#         a = mc.MarkovChain().from_data(tmp)
#         A.append(a)

#     ind = np.where(np.array([len(a.observed_matrix) for a in A])==4)[0]
#     # np.array(A[ind].n_order_matrix)
#     if len(ind)>0:
#         PM.append(np.dstack([np.array(A[i].observed_p_matrix) for i in ind]).mean(axis=-1))
#         O2M.append(np.dstack([np.array(A[i].n_order_matrix(order=2)) for i in ind]).mean(axis=-1))
#         O3M.append(np.dstack([np.array(A[i].n_order_matrix(order=3)) for i in ind]).mean(axis=-1))
#     else:
#         PM.append(np.nan)
#         O2M.append(np.nan)
#         O3M.append(np.nan)


# np.savez('summaries/MR_transition_matrices.npz', MR_PM = PM, MR_O2M = O2M, MR_O3M = O3M)


# ## veg, water, sed, wood
# PM = []; O2M = []; O3M = []
# ## all - instantanues
# for counter,g in tqdm(enumerate(MRbudget_reaches_redo)):
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

# np.savez('summaries/MR_transition_matrices_budgetreaches.npz', MR_PM = PM, MR_O2M = O2M, MR_O3M = O3M)



# ############### MR

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


# ############### LR

# with np.load('summaries/LR_transition_matrices_budgetreaches_partial0_41.npz', allow_pickle=True) as dat:
#     LR_tpm = dict()
#     for k in dat.keys():
#         LR_tpm[k] = dat[k]
#     del dat

# LR_TPM = []
# for k in LR_tpm['LR_PM']:
#     if np.isnan(k).any():
#         tmp = np.ones((4,4))*np.nan
#         LR_TPM.append(tmp)
#     else:
#         LR_TPM.append(k)

# with np.load('summaries/LR_transition_matrices_budgetreaches_partial42_52.npz', allow_pickle=True) as dat:
#     LR_tpm = dict()
#     for k in dat.keys():
#         LR_tpm[k] = dat[k]
#     del dat

# for k in LR_tpm['LR_PM']:
#     if np.isnan(k).any():
#         tmp = np.ones((4,4))*np.nan
#         LR_TPM.append(tmp)
#     else:
#         LR_TPM.append(k)

# #############################

# dMR_TPM = np.dstack(MR_TPM)
# dLR_TPM = np.dstack(LR_TPM)


# plt.figure(figsize=(12,8))
# plt.subplot(221)
# g = sns.heatmap(np.nanmedian(dMR_TPM,axis=-1), annot = True, cmap ='plasma', vmax=1, vmin=0,
#             linecolor ='black', linewidths = 1, cbar_kws={'label': 'Probability of transition'})

# g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
# g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])

# plt.subplot(222)
# g = sns.heatmap(np.nanmedian(dLR_TPM,axis=-1), annot = True, cmap ='plasma', vmax=1, vmin=0,
#             linecolor ='black', linewidths = 1, cbar_kws={'label': 'Probability of transition'})

# g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
# g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])

# plt.savefig("summaries/median_reach_TPM_ds.png", dpi=300, bbox_inches="tight")
# plt.close()


# ############### MR

# MR_wood_pers = [p[3,3] for p in MR_TPM]#
# MR_sed_pers = [p[2,2] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_veg_pers = [p[0,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_water_pers = [p[1,1] for p in  MR_TPM]#MR_tpm['MR_PM']]

# MR_veg_erosion = [p[0,1] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_veg_burial = [p[0,2] for p in MR_TPM]# MR_tpm['MR_PM']]
# MR_canopy_emerg = [p[0,3] for p in  MR_TPM]#MR_tpm['MR_PM']]

# MR_veg_enc = [p[1,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_sed_dep = [p[1,2] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_wood_dep_fromwater = [p[1,3] for p in  MR_TPM]#MR_tpm['MR_PM']]

# MR_veg_growth = [p[2,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_sed_erosion = [p[2,1] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_wood_dep_fromsed = [p[2,3] for p in  MR_TPM]#MR_tpm['MR_PM']]

# MR_wood_occl = [p[3,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_wood_erosion = [p[3,1] for p in  MR_TPM]#MR_tpm['MR_PM']]
# MR_wood_burial = [p[3,2] for p in  MR_TPM]#MR_tpm['MR_PM']]

# MRxy = np.array([np.mean(np.mean(g['coordinates'],axis=0),axis=0).flatten() for g in MRbudget_reaches_redo])
# MRx = MRxy[:,0]
# MRy = MRxy[:,1]



# ############### LR

# LR_wood_pers = [p[3,3] for p in LR_TPM]#
# LR_sed_pers = [p[2,2] for p in  LR_TPM]#LR_tpm['LR_PM']]
# LR_veg_pers = [p[0,0] for p in  LR_TPM]#LR_tpm['LR_PM']]
# LR_water_pers = [p[1,1] for p in  LR_TPM]#LR_tpm['LR_PM']]

# LR_veg_erosion = [p[0,1] for p in  LR_TPM]#LR_tpm['LR_PM']]
# LR_veg_burial = [p[0,2] for p in LR_TPM]# LR_tpm['LR_PM']]
# LR_canopy_emerg = [p[0,3] for p in  LR_TPM]#LR_tpm['LR_PM']]

# LR_veg_enc = [p[1,0] for p in  LR_TPM]#LR_tpm['LR_PM']]
# LR_sed_dep = [p[1,2] for p in  LR_TPM]#LR_tpm['LR_PM']]
# LR_wood_dep_fromwater = [p[1,3] for p in  LR_TPM]#LR_tpm['LR_PM']]

# LR_veg_growth = [p[2,0] for p in  LR_TPM]#LR_tpm['LR_PM']]
# LR_sed_erosion = [p[2,1] for p in  LR_TPM]#LR_tpm['LR_PM']]
# LR_wood_dep_fromsed = [p[2,3] for p in  LR_TPM]#LR_tpm['LR_PM']]

# LR_wood_occl = [p[3,0] for p in  LR_TPM]#LR_tpm['LR_PM']]
# LR_wood_erosion = [p[3,1] for p in  LR_TPM]#LR_tpm['LR_PM']]
# LR_wood_burial = [p[3,2] for p in  LR_TPM]#LR_tpm['LR_PM']]

# LRxy = np.array([np.mean(np.mean(g['coordinates'],axis=0),axis=0).flatten() for g in LRbudget_reaches_redo])
# LRx = LRxy[:,0]
# LRy = LRxy[:,1]



# ##################################################################

# dat = {
#     ## persistence
#     "MR_wood_pers": np.array(MR_wood_pers),
#     "MR_sed_pers": np.array(MR_sed_pers),
#     "MR_veg_pers": np.array(MR_veg_pers),
#     "MR_water_pers": np.array(MR_water_pers),

#     ## remov veg
#     "MR_veg_erosion": np.array(MR_veg_erosion),    
#     "MR_veg_burial": np.array(MR_veg_burial),   

#     ## added veg
#     "MR_veg_enc": np.array(MR_veg_enc),
#     "MR_veg_growth": np.array(MR_veg_growth),

#     ## added wood
#     "MR_wood_dep_fromwater": np.array(MR_wood_dep_fromwater),      
#     "MR_wood_dep_fromsed": np.array(MR_wood_dep_fromsed),  
#     "MR_canopy_emerg": np.array(MR_canopy_emerg),

#     ## sed
#     "MR_sed_erosion": np.array(MR_sed_erosion),    
#     "MR_sed_dep": np.array(MR_sed_dep),      

#     ## remove wood
#     "MR_wood_occl": np.array(MR_wood_occl),
#     "MR_wood_erosion": np.array(MR_wood_erosion),      
#     "MR_wood_burial": np.array(MR_wood_burial),    
# }

# dat = pd.DataFrame.from_dict(dat, orient='index')

# dat2 = dat.T.dropna()

# add_wood = dat2.MR_wood_dep_fromwater.values+ dat2.MR_wood_dep_fromsed.values+ dat2.MR_canopy_emerg.values
# remove_wood = dat2.MR_wood_occl.values+ dat2.MR_wood_erosion.values+ dat2.MR_wood_burial.values

# add_veg = dat2.MR_veg_enc.values+ dat2.MR_veg_growth.values
# remove_veg = dat2.MR_veg_erosion.values+ dat2.MR_veg_burial.values


# MRi  = np.interp(np.linspace(0,len(MR),len(dat2)), np.arange(len(MR)), MR)
# #########################################


# LRdat = {
#     ## persistence
#     "LR_wood_pers": np.array(LR_wood_pers),
#     "LR_sed_pers": np.array(LR_sed_pers),
#     "LR_veg_pers": np.array(LR_veg_pers),
#     "LR_water_pers": np.array(LR_water_pers),

#     ## remov veg
#     "LR_veg_erosion": np.array(LR_veg_erosion),    
#     "LR_veg_burial": np.array(LR_veg_burial),   

#     ## added veg
#     "LR_veg_enc": np.array(LR_veg_enc),
#     "LR_veg_growth": np.array(LR_veg_growth),

#     ## added wood
#     "LR_wood_dep_fromwater": np.array(LR_wood_dep_fromwater),      
#     "LR_wood_dep_fromsed": np.array(LR_wood_dep_fromsed),  
#     "LR_canopy_emerg": np.array(LR_canopy_emerg),

#     ## sed
#     "LR_sed_erosion": np.array(LR_sed_erosion),    
#     "LR_sed_dep": np.array(LR_sed_dep),      

#     ## remove wood
#     "LR_wood_occl": np.array(LR_wood_occl),
#     "LR_wood_erosion": np.array(LR_wood_erosion),      
#     "LR_wood_burial": np.array(LR_wood_burial),    
# }

# LRdat = pd.DataFrame.from_dict(LRdat, orient='index')

# LRdat2 = LRdat.T.dropna()

# LRadd_wood = LRdat2.LR_wood_dep_fromwater.values+ LRdat2.LR_wood_dep_fromsed.values+ LRdat2.LR_canopy_emerg.values
# LRremove_wood = LRdat2.LR_wood_occl.values+ LRdat2.LR_wood_erosion.values+ LRdat2.LR_wood_burial.values

# LRadd_veg = LRdat2.LR_veg_enc.values+ LRdat2.LR_veg_growth.values
# LRremove_veg = LRdat2.LR_veg_erosion.values+ LRdat2.LR_veg_burial.values


# LRi  = np.interp(np.linspace(0,len(LR),len(LRdat2)), np.arange(len(LR)), LR)





# plt.figure(figsize=(12,18))

# plt.subplot(621)
# plt.plot(MRi, add_wood/3,'-', color='brown', label="Processes that add wood)")
# plt.plot(MRi, remove_wood/3,'--', color='brown', label="Wood removal processes")
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(0,1)
# plt.title('a)', loc='left')

# plt.subplot(623)
# plt.plot(MRi, (add_wood/3) - (remove_wood/3), 'k', label="Net probability")
# plt.plot(MRi, dat2.MR_wood_pers.values,':', lw=2, color='brown', label="Persistent wood")
# plt.axhline(0)
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(-1,1)
# plt.title('b)', loc='left')

# plt.subplot(625)
# plt.plot(MRi,dat2.MR_sed_dep.values,'-', color='gold', label="Processes that add sediment)")
# plt.plot(MRi,dat2.MR_sed_erosion.values,'--', color='gold', label="Sediment removal processes")
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(0,1)
# plt.title('c)', loc='left')

# plt.subplot(627)
# plt.plot(MRi, (dat2.MR_sed_dep.values) - (dat2.MR_sed_erosion.values), 'k', label="Net probability")
# plt.plot(MRi, dat2.MR_sed_pers.values,':', lw=2, color='gold', label="Persistent sediment")
# plt.axhline(0)
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(-1,1)
# plt.title('d)', loc='left')

# plt.subplot(629)
# plt.plot(MRi, add_veg/2,'-', color='lawngreen', label="Processes that add veg.)")
# plt.plot(MRi, remove_veg/2,'--', color='lawngreen', label="Veg. removal processes")
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(0,1)
# plt.title('e)', loc='left')

# plt.subplot(6,2,11)
# plt.plot(MRi, (add_veg/2) - (remove_veg/2), 'k', label="Net probability")
# plt.plot(MRi, dat2.MR_veg_pers.values,':', lw=2, color='lawngreen', label="Persistent veg.")
# plt.axhline(0)
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(-1,1)
# plt.xlabel('River kilometer')
# plt.ylabel('Probability')
# plt.title('f)', loc='left')


# #### LR

# plt.subplot(622)
# plt.plot(LRi, LRadd_wood/3,'-', color='brown', label="Processes that add wood)")
# plt.plot(LRi, LRremove_wood/3,'--', color='brown', label="Wood removal processes")
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(0,1)

# plt.subplot(624)
# plt.plot(LRi, (LRadd_wood/3) - (LRremove_wood/3), 'k', label="Net probability")
# plt.plot(LRi, LRdat2.LR_wood_pers.values,':', lw=2, color='brown', label="Persistent wood")
# plt.axhline(0)
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(-1,1)

# plt.subplot(626)
# plt.plot(LRi,LRdat2.LR_sed_dep.values,'-', color='gold', label="Processes that add sediment)")
# plt.plot(LRi,LRdat2.LR_sed_erosion.values,'--', color='gold', label="Sediment removal processes")
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(0,1)

# plt.subplot(628)
# plt.plot(LRi, (LRdat2.LR_sed_dep.values) - (LRdat2.LR_sed_erosion.values), 'k', label="Net probability")
# plt.plot(LRi, LRdat2.LR_sed_pers.values,':', lw=2, color='gold', label="Persistent sediment")
# plt.axhline(0)
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(-1,1)

# plt.subplot(6,2,10)
# plt.plot(LRi, LRadd_veg/2,'-', color='lawngreen', label="Processes that add veg.)")
# plt.plot(LRi, LRremove_veg/2,'--', color='lawngreen', label="Veg. removal processes")
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(0,1)

# plt.subplot(6,2,12)
# plt.plot(LRi, (LRadd_veg/2) - (LRremove_veg/2), 'k', label="Net probability")
# plt.plot(LRi, LRdat2.LR_veg_pers.values,':', lw=2, color='lawngreen', label="Persistent veg.")
# plt.axhline(0)
# plt.legend(fontsize=7)
# plt.gca().invert_xaxis()
# plt.ylim(-1,1)
# plt.xlabel('River kilometer')
# plt.ylabel('Probability')


# # plt.show()
# plt.savefig("summaries/MR_LR_transition_dynamics.png", dpi=300, bbox_inches="tight")
# plt.close()



#############################################################



# cmap = ListedColormap(["black","lawngreen", "blue", "gold", "brown"])
# colors = ["lawngreen", "blue", "gold", "brown"]


# ds =  [str(i)[:4] for i in MRi] 
# width = 0.5

# fig, ax = plt.subplots(nrows = 5, ncols = 1, figsize=(15,20))

# tmp = dat2[["MR_veg_pers", "MR_water_pers", "MR_sed_pers", "MR_wood_pers"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[0].bar(ds, w, width, label=label, bottom=bottom, color=colors[counter])
#     bottom += w
#     counter += 1
# ax[0].set_title("a) Persistence",loc='left')
# ax[0].legend(loc="upper right")
# # ax[0].invert_xaxis()
# ax[0].set_ylim(0,2)

# colors2 = ['cornflowerblue','goldenrod','yellowgreen']
# tmp = dat2[["MR_wood_dep_fromwater", "MR_wood_dep_fromsed", "MR_canopy_emerg"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     p = ax[1].bar(ds, w, width, label=label, bottom=bottom, color=colors2[counter])
#     bottom += w
#     counter += 1
# ax[1].set_title("b) Added wood",loc='left')
# ax[1].legend(loc="upper right")
# # ax[1].invert_xaxis()
# ax[1].set_ylim(0,2)

# colors3 = ['yellowgreen','lightsteelblue','khaki']
# tmp = dat2[["MR_wood_occl", "MR_wood_erosion", "MR_wood_burial"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[2].bar(ds, w, width, label=label, bottom=bottom, color=colors3[counter])
#     bottom += w
#     counter += 1
# ax[2].set_title("c) Removed wood",loc='left')
# ax[2].legend(loc="upper right")
# # ax[2].invert_xaxis()
# ax[2].set_ylim(0,2)

# colors4 = ['deepskyblue','gold']
# tmp = dat2[["MR_sed_erosion", "MR_sed_dep"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[3].bar(ds, w, width, label=label, bottom=bottom, color=colors4[counter])
#     bottom += w
#     counter += 1
# ax[3].set_title("d) Sediment",loc='left')
# ax[3].legend(loc="upper right")
# # ax[3].invert_xaxis()
# ax[3].set_ylim(0,2)

# colors5 = ['deepskyblue','gold', 'palegreen', 'limegreen']
# tmp = dat2[["MR_veg_erosion", "MR_veg_burial", "MR_veg_enc", "MR_veg_growth"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[4].bar(ds, w, width, label=label, bottom=bottom, color=colors5[counter])
#     bottom += w
#     counter += 1
# ax[4].set_title("e) Vegetation",loc='left')
# ax[4].legend(loc="upper right")
# # ax[4].invert_xaxis()
# ax[4].set_ylim(0,2)
# plt.xlabel('River kilometer')
# plt.ylabel('Transition probability')

# # plt.show()
# plt.savefig("summaries/MR_transitions_ds.png", dpi=300, bbox_inches="tight")
# plt.close()






# ds =  [str(i)[:4] for i in LRi] 
# width = 0.5

# fig, ax = plt.subplots(nrows = 5, ncols = 1, figsize=(15,20))

# tmp = LRdat2[["LR_veg_pers", "LR_water_pers", "LR_sed_pers", "LR_wood_pers"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[0].bar(ds, w, width, label=label, bottom=bottom, color=colors[counter])
#     bottom += w
#     counter += 1
# # ax[0].set_title("a) Persistence",loc='left')
# ax[0].legend(loc="upper right")
# # ax[0].invert_xaxis()
# ax[0].set_ylim(0,2)

# colors2 = ['cornflowerblue','goldenrod','yellowgreen']
# tmp = LRdat2[["LR_wood_dep_fromwater", "LR_wood_dep_fromsed", "LR_canopy_emerg"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     p = ax[1].bar(ds, w, width, label=label, bottom=bottom, color=colors2[counter])
#     bottom += w
#     counter += 1
# # ax[1].set_title("b) Added wood",loc='left')
# ax[1].legend(loc="upper right")
# # ax[1].invert_xaxis()
# ax[1].set_ylim(0,2)

# colors3 = ['yellowgreen','lightsteelblue','khaki']
# tmp = LRdat2[["LR_wood_occl", "LR_wood_erosion", "LR_wood_burial"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[2].bar(ds, w, width, label=label, bottom=bottom, color=colors3[counter])
#     bottom += w
#     counter += 1
# # ax[2].set_title("c) Removed wood",loc='left')
# ax[2].legend(loc="upper right")
# # ax[2].invert_xaxis()
# ax[2].set_ylim(0,2)

# colors4 = ['deepskyblue','gold']
# tmp = LRdat2[["LR_sed_erosion", "LR_sed_dep"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[3].bar(ds, w, width, label=label, bottom=bottom, color=colors4[counter])
#     bottom += w
#     counter += 1
# # ax[3].set_title("d) Sediment",loc='left')
# ax[3].legend(loc="upper right")
# # ax[3].invert_xaxis()
# ax[3].set_ylim(0,2)

# colors5 = ['deepskyblue','gold', 'palegreen', 'limegreen']
# tmp = LRdat2[["LR_veg_erosion", "LR_veg_burial", "LR_veg_enc", "LR_veg_growth"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[4].bar(ds, w, width, label=label, bottom=bottom, color=colors5[counter])
#     bottom += w
#     counter += 1
# # ax[4].set_title("e) Vegetation",loc='left')
# ax[4].legend(loc="upper right")
# # ax[4].invert_xaxis()
# ax[4].set_ylim(0,2)
# plt.xlabel('River kilometer')
# plt.ylabel('Transition probability')

# # plt.show()
# plt.savefig("summaries/LR_transitions_ds.png", dpi=300, bbox_inches="tight")
# plt.close()








# gdf = gpd.GeoDataFrame(
#     dat.T, geometry=gpd.points_from_xy(np.array(MRx), np.array(MRy)), crs="EPSG:26911"
# )

# ## write to geojson file, using a wgs84 crs for the geometry
# with open('MR_tpm_processes.geojson' , 'w') as file:
#     file.write(gdf.to_crs("epsg:26911").to_json())




# fig1, ax1 = plt.subplots(nrows=2,ncols=1)
# ax1.pcolormesh(reference['x'].to_numpy(), reference['y'].to_numpy(),  reference['red'][:-1,:-1], cmap='gray')
# ax1.scatter(MRx,MRy,20,MR_wood_erosion , cmap='bwr')
# # plt.show()
# plt.savefig("test.png", dpi=200, bbox_inches='tight')
# plt.close()



# plt.scatter(MRx,MRy,10,MR_wood_erosion )
# plt.show()


# species = ([str(k) for k in range(len(MR_wood_pers))])

# MR_weight_counts = {
#     "Wood": np.array(MR_wood_pers),
#     "Sed": np.array(MR_sed_pers),
#     "Veg": np.array(MR_veg_pers),
#     "Water": np.array(MR_water_pers)
# }

# width = 0.5

# fig, ax = plt.subplots()
# bottom = np.zeros(len(species))

# for boolean, weight_count in MR_weight_counts.items():
#     p = ax.bar(weight_count, width, label=boolean, bottom=bottom)
#     bottom += weight_count

# ax.set_title("Landcover persistence")
# ax.legend(loc="upper right")

# plt.show()

# tmp = np.array([[0.34736165, 0.22265122, 0.36679537, 0.06319176],
#     [0.05255255, 0.62374517, 0.29045474, 0.03324753],
#     [0.04113042, 0.69750107, 0.2202381 , 0.04113042],
#     [0.14864865, 0.36486486, 0.27027027, 0.21621622]])

# g = sns.heatmap(tmp, annot = True, cmap ='plasma', 
#             linecolor ='black', linewidths = 1, cbar_kws={'label': 'Probability of transition'})

# g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
# g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])

# # plt.show()

# plt.savefig("summaries/example_TPM.png", dpi=300, bbox_inches="tight")
# plt.close()


# import json, os
# import rioxarray
# import xarray as xr 
# from glob import glob 
# from dask.distributed import Client
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors
# from scipy import ndimage
# from skimage.exposure import match_histograms

# #############################################################
# #############################################################
# #############################################################
# #################### user inputs 

# dtype = 'float64'
# chunksize = ("auto", "auto")

# times = [
#     '2012-04-07',
#     '2012-08-10',
#     '2012-11-08',
#     '2013-02-13',
#     '2013-04-30',
#     '2013-09-19',
#     '2014-02-01',
#     '2014-09-30',
#     '2015-03-03',
#     '2015-09-23',
#     '2016-01-11',
#     '2016-07-14',
#     '2016-09-30',
#     '2017-09-22'
# ]

# # n_workers = 22
# # threads_per_worker = 2
# # memory_limit='115GB'
# # ## start client
# # client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

# cwd = os.getcwd()

# run_bash = False

# # Create variable used for time axis
# time_var = xr.Variable('time',times)

# #############################################################


# ######### get movie regions and clipper
# movie_regions = sorted(glob('../raw_data/GIS/MR*movie*epsg6339.geojson'))
# print("{} movie_ regions".format(len(movie_regions)))

# movie_geometries = []
# for r in movie_regions:
#     with open(r) as f:
#         gj = json.load(f)
#     features = gj['features'][0]

#     movie_geometries.append(features['geometry'])


# r = "../raw_data/GIS/MR_movie_bars.geojson"
# with open(r) as f:
#     gj = json.load(f)
# MR_bars = [a['geometry'] for a in gj['features']]


# #############################################################
# #########################################################
# ### regrid DEM rasters
# ### recombine (mosaic) and regrid
# # all "results" rasters are 15928 x 41411
# # pixel = 1.569605128802169152e-06 degrees (approx 15cm)
# # gridded to extents of grid.geojson

# if run_bash:
#     os.chdir("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/")
#     os.system("bash regridMR.sh") 
#     os.chdir(cwd)

# #############################################################
# #########################################################
# ## time-series at every point
# veg_files = sorted(glob('../raw_data/MR/MR_veg/MR_*_Prob1_regrid.tif'))
# # water_files = sorted(glob('../raw_data/MR/MR_water/MR_*_Prob0_regrid.tif'))
# # dev_files = sorted(glob('../raw_data/MR/MR_dev/MR_*_Prob1_regrid.tif'))
# dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/*MR_*DEM_regrid.tif'))
# print(len(dem_files))

# sed_files = sorted(glob('../results/MR/MR_sed/Elwha_*sed.tif'))
# print(len(sed_files))

# # wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/Elwha_MR_*wood_filtered_bin0.1_regrid_final.tif'))
# wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/model1/MR_*cleaned.tif'))

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
# print(sed_geotiffs_ds.to_array().shape)

# # get timeaverage image for consistent lighting
# avim_ds = rioxarray.open_rasterio("../results/MR/MR_orthos_orig/Elwha_MR_im_time_mean_prob_regrid.tif", chunks=chunksize, dtype='uint8')
# avim_ds = avim_ds.to_dataset('band')
# print(avim_ds.dims)
# print(wood_geotiffs_ds.dims)


# # ####### MR
# # MR_detrend_dem = rioxarray.open_rasterio("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_global_regrid.tif", chunks=chunksize, dtype=dtype)
# # MR_detrend_dem = MR_detrend_dem.to_dataset('band').persist()
# # print(MR_detrend_dem.dims)

# # dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_2*.tif'))
# # geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
# #                         dim=time_var)
# # # Covert our xarray.DataArray into a xarray.Dataset
# # geotiffs_ds = geotiffs_da.to_dataset('band')
# # # Rename the variable to a more useful name
# # dem_detrend_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})


# # #########################################
# # ################ movies with histogram-matched imagery

# #############################################################
# im_files = sorted(glob('../raw_data/MR/MR_orthos_orig/Elwha_MR_*_regrid.tif'))
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

# #############################################################
# #############################################################

# #############################################################
# #### focused movies


# ### wood chronology - different color per time
# cmap = plt.get_cmap('inferno', len(times))
# custom_palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
# bounds=[0,1]

# ## wood 
# for counter,g in tqdm(enumerate(MR_bars)):
#     print("Working on region {}".format(counter))

#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
#     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
#     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))

#         im_da = tmp_da.sel(time=time).transpose()/255.
#         refim_da = reftmp_da.transpose()/255.
#         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         wood_da = ndimage.maximum_filter(wood_da, size=10)

#         if inner_counter==0:
#             fig1, ax1 = plt.subplots()
#             ax1.imshow(matched)
#         # ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
#         wood_da[wood_da==0] = np.nan
#         # ax1.imshow(wood_da, 'Reds_r', alpha=0.25)

#         # make a color map of fixed colors
#         cmap_tmp = matplotlib.colors.ListedColormap(['white', custom_palette[inner_counter]])
#         norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
#         ax1.imshow(wood_da, interpolation='nearest', cmap=cmap_tmp, norm=norm, alpha=0.5)
#         plt.axis('off')
#         # plt.title(time)

#         # plt.show()
#         plt.savefig(f"../results/MR/MR_Bars_Wood_chron_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         # plt.close()
#         del wood_da

#     plt.close()





# ## wood 
# for counter,g in tqdm(enumerate(MR_bars)):
#     print("Working on region {}".format(counter))


#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
#     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
#     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))

#         im_da = tmp_da.sel(time=time).transpose()/255.
#         refim_da = reftmp_da.transpose()/255.
#         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         wood_da = ndimage.maximum_filter(wood_da, size=10)

#         fig1, ax1 = plt.subplots()
#         ax1.imshow(matched)
#         ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
#         wood_da[wood_da==0] = np.nan
#         ax1.imshow(wood_da, 'Reds_r', alpha=0.25)
#         plt.title(time)
#         plt.axis('off')

#         # plt.show()
#         plt.savefig(f"../results/MR/MR_Bars_Wood_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da




# ## wood + sediment
# for counter,g in tqdm(enumerate(MR_bars)):
#     print("Working on region {}".format(counter))


#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
#     sed_c = sed_geotiffs_ds.rio.clip([g], sed_geotiffs_ds.rio.crs)


#     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
#     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))

#         im_da = tmp_da.sel(time=time).transpose()/255.
#         refim_da = reftmp_da.transpose()/255.
#         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         wood_da = ndimage.maximum_filter(wood_da, size=10)

#         sed_da = sed_c.sed.sel(time=time)
#         sed_da = sed_da.transpose().to_numpy()
#         sed_da[sed_da==0] = np.nan

#         fig1, ax1 = plt.subplots()
#         ax1.imshow(matched)
#         ax1.imshow(sed_da, alpha=0.5, cmap='YlOrRd')

#         ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
#         wood_da[wood_da==0] = np.nan
#         ax1.imshow(wood_da, 'Reds_r', alpha=0.25)
#         plt.title(time)
#         plt.axis('off')

#         # plt.show()
#         plt.savefig(f"../results/MR/MR_Bars_Wood_Sed_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da
#         del sed_da



# ## wood age (sum)
# for counter,g in tqdm(enumerate(MR_bars)):
#     print("Working on region {}".format(counter))

#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
#     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
#     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

#     sum_array = []
#     for inner_counter, time in enumerate(times):
#         # print("Working on time {}".format(time))
#         wood_da = wood_c.wood.sel(time=time)
#         wood_da = wood_da.transpose().to_numpy()
#         wood_da = ndimage.maximum_filter(wood_da, size=10)
#         sum_array.append(wood_da)

#     sum_ = np.cumsum(np.dstack(sum_array),axis=-1)
#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))
#         im_da = tmp_da.sel(time=time).transpose()/255.
#         refim_da = reftmp_da.transpose()/255.
#         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

#         fig1, ax1 = plt.subplots()
#         ax1.imshow(matched)

#         ax1.imshow(sum_[:,:,inner_counter], 'inferno', alpha=0.5)
#         plt.title(time)
#         plt.axis('off')

#         # plt.show()
#         plt.savefig(f"../results/MR/MR_Bars_Wood_age_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()



# ## sediment age (sum)
# for counter,g in tqdm(enumerate(MR_bars)):
#     print("Working on region {}".format(counter))

#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     sed_c = sed_geotiffs_ds.rio.clip([g], sed_geotiffs_ds.rio.crs)
#     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
#     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

#     sum_array = []
#     for inner_counter, time in enumerate(times):
#         # print("Working on time {}".format(time))
#         sed_da = sed_c.sed.sel(time=time)
#         sed_da = sed_da.transpose().to_numpy()
#         sum_array.append(sed_da)

#     sum_ = np.cumsum(np.dstack(sum_array),axis=-1)
#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))
#         im_da = tmp_da.sel(time=time).transpose()/255.
#         refim_da = reftmp_da.transpose()/255.
#         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

#         fig1, ax1 = plt.subplots()
#         ax1.imshow(matched)

#         ax1.imshow(sum_[:,:,inner_counter], 'inferno', alpha=0.5)
#         plt.title(time)
#         plt.axis('off')

#         # plt.show()
#         plt.savefig(f"../results/MR/MR_Bars_Sed_age_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()


# # #############################################################
# # #### focused movies with HEIGHT contours

# # # This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
# # # then adds a percent sign.
# # def fmt(x):
# #     s = f"{x:.1f}"
# #     if s.endswith("0"):
# #         s = f"{x:.0f}"
# #     return s


# # ## wood 
# # for counter,g in tqdm(enumerate(MR_bars)):
# #     print("Working on region {}".format(counter))


# #     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
# #     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
# #     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
# #     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
# #     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))
# #     dem_c = dem_detrend_geotiffs_ds.rio.clip([g], dem_detrend_geotiffs_ds.rio.crs)

# #     for inner_counter, time in enumerate(times):
# #         print("Working on time {}".format(time))

# #         im_da = tmp_da.sel(time=time).transpose()/255.
# #         refim_da = reftmp_da.transpose()/255.
# #         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
# #         wood_da = wood_c.wood.sel(time=time)
# #         wood_da = wood_da.transpose().to_numpy()
# #         wood_da = ndimage.maximum_filter(wood_da, size=10)
# #         dem_da = dem_c.dem.sel(time=time)
# #         dem_da = dem_da.transpose().to_numpy()

# #         fig1, ax1 = plt.subplots()
# #         ax1.imshow(matched)
# #         ax1.contour(wood_da, colors='k') #,[-99,0,99], color='r')#, 
# #         wood_da[wood_da==0] = np.nan
# #         ax1.imshow(wood_da, 'Reds_r', alpha=0.25)
# #         plt.title(time)
# #         plt.axis('off')

# #         ## add DEM contours
# #         CS = ax1.contour(dem_da)
# #         ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=5)

# #         # plt.show()
# #         plt.savefig(f"../results/MR/MR_Bars_Wood_Contours_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
# #         plt.close()
# #         del wood_da








# # ## movie_geometries
# # ## wood-sum over static image
# # for counter,g in tqdm(enumerate(movie_geometries)):
# #     print("Working on region {}".format(counter))

# #     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)

# #     im_c = avim_ds.rio.clip([g], avim_ds.rio.crs)
# #     tmp_da = xr.concat([im_c[1],im_c[2],im_c[3]],dim=('x','x','x'))
# #     del im_c

# #     fig1, ax1 = plt.subplots()
# #     ax1.imshow(tmp_da.transpose()/255.)
# #     del tmp_da
    
# #     wood_da = wood_c.wood.sum("time", skipna=True).to_numpy()
# #     wood_da[wood_da==0] = np.nan
# #     im=ax1.imshow(wood_da.transpose()/len(times),cmap='Reds_r', vmin=0, vmax=1)
# #     plt.colorbar(im, shrink=0.5)
# #     del wood_da, wood_c
# #     # plt.show()

# #     plt.axis('off')
# #     plt.savefig(f"../results/MR/Wood_woodsum_inst_movie_{counter}.png", dpi=300, bbox_inches='tight')
# #     plt.close()


# ##### all
# for counter,g in tqdm(enumerate(MR_bars)):
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
#         plt.savefig(f"../results/MR/All_inst_movie_bars_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da, water_da, veg_da, matched #dem_da, 




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
#         plt.savefig(f"../results/MR/Wood_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del wood_da



# # ##### all regions

# # ## wood only
# # for counter,g in tqdm(enumerate(geometries)):
# #     print("Working on region {}".format(counter))

# #     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
# #     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
# #     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
# #     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
# #     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

# #     for inner_counter, time in enumerate(times):
# #         print("Working on time {}".format(time))

# #         fig1, ax1 = plt.subplots()

# #         im_da = tmp_da.sel(time=time).transpose()/255.
# #         refim_da = reftmp_da.transpose()/255.

# #         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

# #         ax1.imshow(matched)

# #         wood_da = wood_c.wood.sel(time=time)
# #         wood_da = wood_da.transpose().to_numpy()
# #         wood_da = ndimage.maximum_filter(wood_da, size=10)
# #         wood_da[wood_da==0] = np.nan
# #         ax1.imshow(wood_da,'Reds_r')
# #         plt.title(time)
# #         # plt.show()

# #         plt.axis('off')
# #         plt.savefig(f"../results/MR/Wood_inst_01_frame_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
# #         plt.close()
# #         del wood_da


# # ##### all
# # for counter,g in tqdm(enumerate(geometries)):
# #     print("Working on region {}".format(counter))

# #     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
# #     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
# #     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
# #     veg_c = veg_geotiffs_ds.rio.clip([g], veg_geotiffs_ds.rio.crs)
# #     water_c = water_geotiffs_ds.rio.clip([g], water_geotiffs_ds.rio.crs)
# #     dem_c = dem_geotiffs_ds.rio.clip([g], dem_geotiffs_ds.rio.crs)
# #     sed_c = sed_geotiffs_ds.rio.clip([g], sed_geotiffs_ds.rio.crs)

# #     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
# #     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

# #     for inner_counter, time in enumerate(times):
# #         print("Working on time {}".format(time))

# #         fig1, ax1 = plt.subplots()

# #         im_da = tmp_da.sel(time=time).transpose()/255.
# #         refim_da = reftmp_da.transpose()/255.

# #         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)

# #         ax1.imshow(matched)

# #         water_da = water_c.water.sel(time=time)
# #         water_da = water_da.transpose().to_numpy()
# #         water_da = ndimage.maximum_filter(water_da, size=10)
# #         water_da[water_da<.2] = np.nan
# #         ax1.imshow(water_da,'Blues', alpha=0.5)

# #         veg_da = veg_c.veg.sel(time=time)
# #         veg_da = veg_da.transpose().to_numpy()
# #         veg_da = ndimage.maximum_filter(veg_da, size=10)
# #         veg_da[veg_da<.5] = np.nan        
# #         ax1.imshow(veg_da,'Purples', alpha=0.5)

# #         sed_da = sed_c.sed.sel(time=time)
# #         sed_da = sed_da.transpose().to_numpy().astype('float64')
# #         sed_da[sed_da<.5] = np.nan        
# #         ax1.imshow(sed_da,'autumn_r', alpha=0.5)

# #         wood_da = wood_c.wood.sel(time=time)
# #         wood_da = wood_da.transpose().to_numpy()
# #         wood_da = ndimage.maximum_filter(wood_da, size=10)
# #         wood_da[wood_da==0] = np.nan
# #         ax1.imshow(wood_da,'Reds_r')

# #         dem_da = dem_c.dem.sel(time=time)

# #         CS1 = ax1.contour(dem_da.transpose(), levels=8, cmap='Greys', alpha=0.5)
# #         ax1.clabel(CS1, CS1.levels[1::2], inline=True, fontsize=5)
# #         plt.title(time)

# #         # plt.show()

# #         plt.axis('off')
# #         plt.savefig(f"../results/MR/All_inst_01_frame_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
# #         plt.close()
# #         del wood_da, dem_da, water_da, veg_da




# # # #########################################
# # # ################ movies with time-averaged imagery


# # #############################################################
# # # cmap=plt.cm.get_cmap('YlOrBr', len(times))
# # # custom_palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

# # for counter,g in tqdm(enumerate(geometries)):
# #     print("Working on region {}".format(counter))

# #     im_c = avim_ds.rio.clip([g], avim_ds.rio.crs)
# #     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
# #     veg_c = veg_geotiffs_ds.rio.clip([g], veg_geotiffs_ds.rio.crs)
# #     water_c = water_geotiffs_ds.rio.clip([g], water_geotiffs_ds.rio.crs)
# #     dem_c = dem_geotiffs_ds.rio.clip([g], dem_geotiffs_ds.rio.crs)

# #     tmp_da = xr.concat([im_c[1],im_c[2],im_c[3]],dim=('x','x','x'))

# #     for inner_counter, time in enumerate(times):
# #         print("Working on time {}".format(time))

# #         fig1, ax1 = plt.subplots()
# #         ax1.imshow(tmp_da.transpose()/255.)

# #         water_da = water_c.water.sel(time=time)
# #         water_da = water_da.transpose().to_numpy()
# #         water_da[water_da<.2] = np.nan
# #         ax1.imshow(water_da,'Blues', alpha=0.5)

# #         veg_da = veg_c.veg.sel(time=time)
# #         veg_da = veg_da.transpose().to_numpy()
# #         veg_da[veg_da<.5] = np.nan        
# #         ax1.imshow(veg_da,'Greens', alpha=0.5)

# #         sed_da = sed_c.sed.sel(time=time)
# #         sed_da = sed_da.transpose().to_numpy().astype('float64')
# #         sed_da[sed_da<.5] = np.nan        
# #         ax1.imshow(sed_da,'autumn_r', alpha=0.5)

# #         wood_da = wood_c.wood.sel(time=time)
# #         wood_da = wood_da.transpose().to_numpy()
# #         wood_da[wood_da==0] = np.nan
# #         ax1.imshow(wood_da,'Reds_r')

# #         # sed_da = np.zeros((wood_c.dims['x'],wood_c.dims['y']))
# #         # sed_da[np.isnan(wood_da) & np.isnan(water_da) & np.isnan(veg_da)] = 1
# #         # sed_da[sed_da==0] = np.nan
# #         # ax1.imshow(sed_da,'YlGn')

# #         dem_da = dem_c.dem.sel(time=time)

# #         CS1 = ax1.contour(dem_da.transpose(), levels=5, cmap='YlOrBr', alpha=0.5)
# #         ax1.clabel(CS1, CS1.levels, inline=True, fontsize=5) #[1::2]
# #         plt.title(time)

# #         # plt.show()

# #         plt.axis('off')
# #         plt.savefig(f"../results/MR/All_01_frame_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
# #         plt.close()
# #         del wood_da, dem_da, water_da, veg_da




# # #########################################
# # ################ bin by elevation

# # # x=np.array(points)[:,0]
# # # y=np.array(points)[:,1]

# # # dat_wood = np.zeros((len(x),len(times)))
# # # dat_water = np.zeros((len(x),len(times)))
# # # dat_veg = np.zeros((len(x),len(times)))
# # # dat_dem = np.zeros((len(x),len(times)))

# # # dem_geotiffs_ds.sel(time=times[0]).min().compute()

# # # x=x[:100]
# # # y=y[:100]

# # # for counter, (xx,yy) in tqdm(enumerate(zip(x,y))):
# # #     # pwood = wood_geotiffs_ds.wood.sel(x=xx,y=yy, method="nearest")
# # #     # pwater = water_geotiffs_ds.water.sel(x=xx,y=yy, method="nearest")
# # #     # pveg = veg_geotiffs_ds.veg.sel(x=xx,y=yy, method="nearest")
# # #     pdem = dem_geotiffs_ds.dem.sel(x=xx,y=yy, method="nearest")
# # #     print(pdem.to_numpy())

# # #     dat_wood[counter,:] = pwood
# # #     dat_water[counter,:] = pwater
# # #     dat_veg[counter,:] = pveg
# # #     dat_dem[counter,:] = pdem

# # # np.savez('../results/MR/MR_wood/summary/bin_wood_water_veg_dem_allpts_5m.npz', dat_veg=dat_veg, dat_water=dat_water, dat_wood=dat_wood, dat_dem=dat_dem, x=x, y=y)


# # #########################################
# # ################ distance to nearest braid

