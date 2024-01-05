
## Dan Buscombe, Marda Science
## 2023

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
import geopandas as gpd
from area import area
from matplotlib.patches import Rectangle


def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    Code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn


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


## open transects file

transfile = '../raw_data/GIS/Elwha_Extended_Transects_shp/extended-transects-MR-LR.geojson'
with open(transfile) as f:
    gj = json.load(f)
transects_all = gj['features']
transects_all_gdf = gpd.GeoDataFrame.from_features(transects_all)

## clip transects by MR externts
MRclipper = gpd.GeoDataFrame.from_features(MRbudget_reaches)
transects_MR = gpd.clip(transects_all_gdf, MRclipper.total_bounds)

## clip transects by LR extents
LRclipper = gpd.GeoDataFrame.from_features(LRbudget_reaches)
transects_LR = gpd.clip(transects_all_gdf, LRclipper.total_bounds)


# xxx, yyy = 

#     for (xx,yy) in tqdm(enumerate(zip(xxx,yyy))):
#         pwood.append(wood_geotiffs_ds.wood.sel(x=xx,y=yy, method="nearest").to_numpy())


#####################################################################

dists = pd.read_csv('br_dists.csv')
LR = np.hstack((0,np.array(dists['LR'])))
MR = np.hstack((0,np.array(dists['MR'][:43])))

## rescale distances
LR = rescale_array(LR,11,2)
MR = rescale_array(MR[::-1],12,20)


#############################################################
#############################################################

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

#############################################################
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


########################################

grid2sqm = 64

# MR_wood_tot = []
# for counter,g in tqdm(enumerate(transects_MR['geometry'])):
#     # print("Working on region {}".format(counter))
#     # print(g)
#     tmp_tot = []
#     for time in times:
#         try:
#             trans_wood = MRwood_geotiffs_ds.sel(time=time).rio.clip([g], MRwood_geotiffs_ds.rio.crs)
#             tot = float(trans_wood.wood.sum().compute().to_numpy())
#             tmp_tot.append(tot)
#         except:
#             tot=np.nan
#             tmp_tot.append(tot)
#     print(tmp_tot)
#     MR_wood_tot.append(tmp_tot)

# LR_wood_tot = []
# for counter,g in tqdm(enumerate(transects_LR['geometry'])):
#     # print("Working on region {}".format(counter))
#     # print(g)
#     tmp_tot = []
#     for time in times:
#         try:
#             trans_wood = LRwood_geotiffs_ds.sel(time=time).rio.clip([g], LRwood_geotiffs_ds.rio.crs)
#             tot = float(trans_wood.wood.sum().compute().to_numpy())
#             tmp_tot.append(tot)
#         except:
#             tot=np.nan
#             tmp_tot.append(tot)
#     print(tmp_tot)
#     LR_wood_tot.append(tmp_tot)


# np.savez('summaries/transects_spacetime_series.npz', LR_wood_tot = LR_wood_tot, MR_wood_tot = MR_wood_tot, times=times, grid2sqm=grid2sqm)


with np.load('summaries/transects_spacetime_series.npz', allow_pickle=True) as f:
    MR_wood_tot = f['MR_wood_tot']
    LR_wood_tot = f['LR_wood_tot']


MR_wood_totarr = np.vstack(MR_wood_tot).reshape(len(times),-1)/grid2sqm

LR_wood_totarr = np.vstack(LR_wood_tot).reshape(len(times),-1)/grid2sqm



with np.load('summaries/Wood_time_series.npz', allow_pickle=True) as f:
    LR_BRarr = f['LR_BRarr']
    MR_BRarr = f['MR_BRarr']
    dt = f['dt']
    grid2sqm = f['grid2sqm']

# ########################################
# plt.figure(figsize=(14,20))
# plt.subplots_adjust(wspace=0.2, hspace=0.2)

# plt.subplot(421)
# plt.imshow(np.flipud(MR_wood_totarr), cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0)
# cb=plt.colorbar(); cb.set_label(r"Wood area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('a)', loc='left')
# # plt.xlabel("Distance downstream (km)"); 

# plt.subplot(422)
# plt.imshow(np.flipud(LR_wood_totarr), cmap='inferno', extent=[LR[0] , LR[-1], dt[0], dt[-1]], aspect='auto', vmin=0)
# cb=plt.colorbar(); cb.set_label(r"Wood area, m$^2$")
# plt.gca().invert_yaxis()

# plt.show()



plt.figure(figsize=(12,12))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.subplot(221)
plt.plot(dt, np.sum(MR_wood_totarr,axis=1),'k-', label='MR', lw=2)
plt.plot(dt, np.sum(LR_wood_totarr,axis=1),'r--', label='LR', lw=2)

plt.show()





MRlocs = np.interp(np.arange(MR_wood_totarr.shape[1]),np.arange(len(MR)),MR)
LRlocs = np.interp(np.arange(LR_wood_totarr.shape[1]),np.arange(len(LR)),LR)

LRlocs = rescale_array(LRlocs,10.1,1.4)
MRlocs = rescale_array(MRlocs,12.3,20.6)

# ########################################

fig, ax1 = plt.subplots(nrows=1,ncols=2, figsize=(12,6))#,sharex=True, sharey=True)

ax1[0].plot(MR[::-1], np.sum(MR_BRarr,axis=0)[::-1],'k',lw=2)
ax1[0].set_ylabel(r'Sub-reach wood area m$^2$', color='k')
ax1[0].tick_params(axis='y', color='k', labelcolor='k')
ax1[0].set_ylim(0,45000)
ax1[0].invert_xaxis()
ax1[0].set_title('a)', loc='left')
ax1[0].set_xlabel(r'River kilometer', color='k')

ax2 = ax1[0].twinx()
ax2.plot(MRlocs[::-1], np.sum(MR_wood_totarr,axis=0)[::-1],'b-o')
ax2.set_ylabel(r'Transect wood area m$^2$', color='b')
ax2.set_ylim(0,24)

ax2.tick_params(axis='y', color='b', labelcolor='b')
ax2.spines['right'].set_color('b')
ax2.spines['left'].set_color('b')

ax1[1].plot(LR[::-1], np.sum(LR_BRarr,axis=0)[::-1],'r--',lw=2)
ax1[1].set_ylabel(r'Sub-reach wood area m$^2$', color='r')
ax1[1].tick_params(axis='y', color='r', labelcolor='r')
ax1[1].set_ylim(0,45000)
ax1[1].invert_xaxis()

y = np.sum(LR_wood_totarr[2:,:],axis=0)[::-1]
y[0:2] = y[0:2]/2
y[0:9] = y[0:9]/2
ax3 = ax1[1].twinx()
ax3.plot(LRlocs[::-1], y,'b-o')
ax3.set_ylabel(r'Transect wood area m$^2$', color='b')
ax3.set_ylim(0,24)

ax3.tick_params(axis='y', color='b', labelcolor='b')
ax3.spines['right'].set_color('b')
ax3.spines['left'].set_color('b')

fig.legend(['Sub-reach','Transect'], bbox_to_anchor=(0.9, 0.85))

# plt.show()
plt.savefig("summaries/transect_subreach_space.png", dpi=300, bbox_inches="tight")
plt.close()
