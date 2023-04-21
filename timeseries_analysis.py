## Dan Buscombe, Marda Science
## Apr, 2023
##
## Does this:
## 1. 
## 2. 
## 3. 
## 4. 
## 5. 
## 6. 

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
# run_bash = True

## factor that converts grid uints 1/8 x 1/8
# into units 1 x 1, i.e. 8 x 8
grid2sqm = 64

## we estimate over-ditizization factor
overdig_factor = 1.2

# #############################################################
# ## start client
# client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

# # Create variable used for time axis
# time_var = xr.Variable('time',times)

# ######### get regions and clipper
# # regions = sorted(glob('../raw_data/GIS/LR*ID*_epsg6339.geojson'))
# # regions = [r for r in regions if 'pts' not in r]
# # print("{} regions".format(len(regions)))

# # geometries = []
# # for r in regions:
# #     with open(r) as f:
# #         gj = json.load(f)
# #     features = gj['features'][0]

# #     geometries.append(features['geometry'])

dt = [datetime.strptime(time,'%Y-%m-%d') for time in times]

brfile = '../results/LR/LR_wood/wood_detect/LR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
LRbudget_reaches = gj['features']

brfile = '../results/MR/MR_wood/wood_detect/MR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
MRbudget_reaches = gj['features']

LRbudget_reaches_redo = []
for b in LRbudget_reaches:
    LRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))

MRbudget_reaches_redo = []
for b in MRbudget_reaches:
    MRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))



brfile = '../results/LR/LR_wood/wood_detect/LR_budget_reaches_epsg4326.geojson'
with open(brfile) as f:
    gj = json.load(f)
LRbudget_reaches2 = gj['features']

brfile = '../results/MR/MR_wood/wood_detect/MR_budget_reaches_epsg4326.geojson'
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



# #############################################################
# #############################################################

# #############################################################
# ### LR
# # get filtered wood probs, clipped to margins
# wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/Elwha_LR_*wood_filtered_bin0.1_regrid_final.tif'))
# print(len(wood_files))

# # Load in and concatenate all individual GeoTIFFs
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')

# # Rename the variable to a more useful name
# LRwood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

# #############################################################
# ### MR
# # get filtered wood probs, clipped to margins
# wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/Elwha_MR_*wood_filtered_bin0.1_regrid_final.tif'))
# print(len(wood_files))

# # Load in and concatenate all individual GeoTIFFs
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')

# # Rename the variable to a more useful name
# MRwood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})


#######################################################

# MR_BR=[]
# for time in times:
#     tmp = MRwood_geotiffs_ds.wood.sel(time=time)
#     for g in tqdm(MRbudget_reaches_redo):
#         wood_c = tmp.rio.clip([g], tmp.rio.crs)
#         result = (wood_c).sum().compute().to_numpy() 
#         MR_BR.append(float(result))

# #######################################################

# LR_BR=[]
# for time in times:
#     tmp = LRwood_geotiffs_ds.wood.sel(time=time)
#     for g in tqdm(LRbudget_reaches_redo):
#         wood_c = tmp.rio.clip([g], tmp.rio.crs)
#         result = (wood_c).sum().compute().to_numpy() 
#         LR_BR.append(float(result))


# LR_BRarr = np.vstack(LR_BR).reshape(len(times),-1)/grid2sqm
# MR_BRarr = np.vstack(MR_BR).reshape(len(times),-1)/grid2sqm

# np.savez('Wood_time_series.npz', LR_BRarr = LR_BRarr, MR_BRarr = MR_BRarr, times=times, dt=dt, grid2sqm=grid2sqm)

with np.load('summaries/Wood_time_series.npz', allow_pickle=True) as f:
    LR_BRarr = f['LR_BRarr']
    MR_BRarr = f['MR_BRarr']
    dt = f['dt']
    grid2sqm = f['grid2sqm']

dists = pd.read_csv('br_dists.csv')
LR = np.hstack((0,np.array(dists['LR'])))
MR = np.hstack((0,np.array(dists['MR'][:43])))

########################################
plt.figure(figsize=(16,16))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(321)
plt.imshow(MR_BRarr, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('a) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

plt.subplot(322)
plt.imshow(LR_BRarr, cmap='viridis', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('b) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

plt.subplot(323)
plt.imshow(np.cumsum(MR_BRarr.T,axis=0).T, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('c) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

plt.subplot(324)
plt.imshow(np.cumsum(LR_BRarr.T,axis=0).T, cmap='viridis', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('d) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

plt.subplot(325)
plt.plot(MR, np.sum(MR_BRarr,axis=0),'k-', label='MR')
plt.plot(LR, np.sum(LR_BRarr,axis=0),'r--', label='LR')
plt.ylabel(r"Sum of estimated wood, m$^2$"); plt.xlabel("Distance downstream (km)"); 
plt.legend()
plt.title('e) ', loc='left')

plt.subplot(326)
plt.plot(dt,np.sum(MR_BRarr,axis=1),'k-', label='MR')
plt.plot(dt,np.sum(LR_BRarr,axis=1),'r--', label='LR')
plt.ylabel(r"Sum of estimated wood, m$^2$");
plt.legend()
plt.title('f) ', loc='left')
# plt.show()

plt.savefig("wood_spacetime_plots.png", dpi=300, bbox_inches="tight")
plt.close()


#### divide out by area of each BR for a wood concentration\
A_MR = np.array(A_MR)
A_LR = np.array(A_LR)

MR_BRarr_c = MR_BRarr/A_MR

LR_BRarr_c = LR_BRarr/A_LR

########################################
plt.figure(figsize=(16,16))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(321)
plt.imshow(MR_BRarr_c, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('a) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

plt.subplot(322)
plt.imshow(LR_BRarr_c, cmap='viridis', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('b) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

plt.subplot(323)
plt.imshow(np.cumsum(MR_BRarr_c.T,axis=0).T, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('c) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

plt.subplot(324)
plt.imshow(np.cumsum(LR_BRarr_c.T,axis=0).T, cmap='viridis', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('d) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

plt.subplot(325)
plt.plot(MR, np.sum(MR_BRarr,axis=0)/(A_MR*len(times)),'k-', label='MR')
plt.plot(LR, np.sum(LR_BRarr,axis=0)/(A_LR*len(times)),'r--', label='LR')
plt.ylabel(r"Wood concentration, m$^2$/m$^2$"); plt.xlabel("Distance downstream (km)"); 
plt.legend()
plt.title('e) ', loc='left')

plt.subplot(326)
plt.plot(dt,np.sum(MR_BRarr,axis=1)/np.sum(A_MR),'k-', label='MR')
plt.plot(dt,np.sum(LR_BRarr,axis=1)/np.sum(A_MR),'r--', label='LR')
plt.ylabel(r"Wood concentration, m$^2$/m$^2$");
plt.legend()
plt.title('f) ', loc='left')
# plt.show()

plt.savefig("wood_dens_spacetime_plots.png", dpi=300, bbox_inches="tight")
plt.close()

##############################################################

sed_load = pd.read_csv('../raw_data/time_series/Elwha_DailySedimentLoads_2011to2016.csv')

sed_load = sed_load[['Day',
'Daily Discharge (m3/s)',
'Total sediment discharge (tonnes)',
'Ave fraction fines (based on two turbidimeters)']]


dt_sed = [datetime.strptime(time,'%m/%d/%Y') for time in sed_load['Day']]
dt_sed = np.array(dt_sed)

ind = np.argsort(dt_sed)


########################################
plt.figure(figsize=(8,8))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(311)
plt.plot(dt_sed[ind], sed_load['Daily Discharge (m3/s)'][ind],'k-')
plt.ylabel(r'Discharge (m$^3$/s)')
plt.title('a) ', loc='left')

plt.subplot(312)
plt.plot(dt_sed[ind], sed_load['Total sediment discharge (tonnes)'][ind],'k-')
plt.ylabel(r'Total sediment discharge (tonnes)')
plt.title('b) ', loc='left')

plt.subplot(313)
plt.plot(dt_sed[ind], sed_load['Ave fraction fines (based on two turbidimeters)'][ind],'k-')
plt.ylabel(r'Average fraction of fines')
plt.title('c) ', loc='left')

# plt.show()
plt.savefig("flow_sed_2011_2016.png", dpi=300, bbox_inches="tight")
plt.close()


########################################
plt.figure(figsize=(24,4))
plt.subplots_adjust(wspace=0.5, hspace=0.3)

plt.subplot(151)
plt.plot(dt_sed[ind], sed_load['Daily Discharge (m3/s)'][ind],'k-')
plt.ylabel(r'Discharge (m$^3$/s)')
plt.title('a) ', loc='left')

plt.subplot(152)
plt.plot(dt_sed[ind], sed_load['Total sediment discharge (tonnes)'][ind],'k-')
plt.ylabel(r'Total sediment discharge (tonnes)')
plt.title('b) ', loc='left')

plt.subplot(153)
plt.plot(dt_sed[ind], sed_load['Ave fraction fines (based on two turbidimeters)'][ind],'k-')
plt.ylabel(r'Average fraction of fines')
plt.title('c) ', loc='left')

plt.subplot(154)
plt.plot(dt,np.sum(MR_BRarr,axis=1)/np.sum(A_MR),'k-', label='MR')
plt.plot(dt,np.sum(LR_BRarr,axis=1)/np.sum(A_MR),'r--', label='LR')
plt.ylabel(r"Estimated wood, m$^2$")
plt.title('d) ', loc='left')

t_sed = np.array([float(d.strftime('%s')) for d in dt_sed[ind]])
t =  np.array([float(d.strftime('%s')) for d in dt])
plt.subplot(155)
plt.plot(np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values), np.sum(MR_BRarr,axis=1), 'ko')

O = np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values)
E = np.sum(MR_BRarr,axis=1)

A = np.vstack([O, np.ones(len(O))]).T
m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
plt.plot(O, m*O+ c, 'r:',lw=2, label='y = '+str(m)[:4]+'x+'+str(c)[:4])
plt.text(20,35000,r'R$^2$ = '+str(np.min(np.corrcoef(O,E))**2)[:6])

plt.ylabel(r"Estimated wood, m$^2$")
plt.xlabel(r'Discharge, day of aerial survey  (m$^3$/s)')
plt.title('e) ', loc='left')

plt.savefig("flow_sed_2011_2016_wood_rel.png", dpi=300, bbox_inches="tight")
plt.close()


# MR_BRarr = np.load('MR_eval_wood_budget.npz')
# #MRtarget_gt_20120407 = target_gt_20120407, MRtarget_gt_20170922=target_gt_20170922, 
# # estMR_BR2=estMR_BR2, estMR_BR=estMR_BR, MRbudget_reaches=MRbudget_reaches_redo, grid2sqm=grid2sqm, overdig_factor=overdig_factor

# LR_BRarr =np.savez('LR_eval_wood_budget.npz')
# # LRtarget_gt_20120407 = target_gt_20120407, LRtarget_gt_20170922=target_gt_20170922, 
# # estLR_BR2=estBR2, estLR_BR=estBR, LRbudget_reaches=budget_reaches_redo, grid2sqm=grid2sqm, overdig_factor=overdig_factor)


# with np.load('../results/LR/LR_wood/summary/LR_wood_water_veg_dem_dist_allpts_5m.npz') as f:
#     dat_veg = f['dat_veg']
#     dat_water = f['dat_water']
#     dat_wood = f['dat_wood']
#     dat_x = f['x']
#     dat_y = f['y']


# # fpoints = sorted(glob('../raw_data/GIS/LR_allpts_clipped_active_10m_wgs84.geojson'))
# # fpoints = sorted(glob('../raw_data/GIS/LR_allpts_clipped_active_5m.geojson'))
# fpoints = sorted(glob('../raw_data/GIS/LR_allpts_clipped_active_10m_v2_wgs84.geojson'))

# with open(fpoints[0]) as f:
#     gj = json.load(f)
# features = gj['features']

# points = [f['geometry']['coordinates'][0] for f in features]
# print("{} sample points".format(len(points)))

# #############################################################
# #########################################################
# ## time-series at every point
# veg_files = sorted(glob('../raw_data/LR/LR_veg/LR_*_Prob1_regrid.tif'))
# water_files = sorted(glob('../raw_data/LR/LR_water/LR_*_Prob0_regrid.tif'))
# # dev_files = sorted(glob('../raw_data/LR/LR_dev/LR_*_Prob1_regrid.tif'))
# dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/*LR_*DEM_regrid.tif'))
# print(len(dem_files))

# # get filtered wood probs, clipped to margins
# wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/Elwha_LR_*bin0.1_regrid_ccc.tif'))

# print(len(wood_files))


# dist_files = sorted(glob('../results/LR/LR_dist2braid/*LR_*.tif'))
# print(len(dist_files))


# # Load in and concatenate all individual GeoTIFFs
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# wood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

# #############################################################
# # Load in and concatenate all individual GeoTIFFs
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in water_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# water_geotiffs_ds = geotiffs_ds.rename({1: 'water'})

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

# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dist_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')
# # Rename the variable to a more useful name
# dist_geotiffs_ds = geotiffs_ds.rename({1: 'dist'})

# #########################################################
# ## clean up
# water_geotiffs_ds = water_geotiffs_ds.drop_vars(2)
# veg_geotiffs_ds = veg_geotiffs_ds.drop_vars(2)
# wood_geotiffs_ds = wood_geotiffs_ds.drop_vars(2)
# dem_geotiffs_ds = dem_geotiffs_ds.drop_vars(2)

# print(water_geotiffs_ds.to_array().shape)
# print(veg_geotiffs_ds.to_array().shape)
# print(wood_geotiffs_ds.to_array().shape)
# print(dem_geotiffs_ds.to_array().shape)
# print(dist_geotiffs_ds.to_array().shape)

# #############################################################
# #########################################################

# x=np.array(points)[:,0]
# y=np.array(points)[:,1]

# print(len(x))


# xx = np.split(x, 13)
# yy = np.split(y, 13)

# def get_pp_stats(xxx,yyy,wood_geotiffs_ds,water_geotiffs_ds,veg_geotiffs_ds,dem_geotiffs_ds):
#     pwood = []; pwater = []; pveg = []; pdem = []; 
#     for (xx,yy) in tqdm(enumerate(zip(xxx,yyy))):
#         pwood.append(wood_geotiffs_ds.wood.sel(x=xx,y=yy, method="nearest").to_numpy())
#         pwater.append(water_geotiffs_ds.water.sel(x=xx,y=yy, method="nearest").to_numpy())
#         pveg.append(veg_geotiffs_ds.veg.sel(x=xx,y=yy, method="nearest".to_numpy()))
#         pdem.append(dem_geotiffs_ds.dem.sel(x=xx,y=yy, method="nearest").to_numpy())
#     return pwood,pwater,pveg,pdem

# pwood,pwater,pveg,pdem = get_pp_stats(xx[0][:100],yy[0][:100],wood_geotiffs_ds,water_geotiffs_ds,veg_geotiffs_ds,dem_geotiffs_ds)

# dat_wood = np.zeros((len(x),len(times)))
# dat_water = np.zeros((len(x),len(times)))
# dat_veg = np.zeros((len(x),len(times)))
# dat_dem = np.zeros((len(x),len(times)))
# dat_dist = np.zeros((len(x),len(times)))

# for counter, (xx,yy) in tqdm(enumerate(zip(x,y))):
#     pwood = wood_geotiffs_ds.wood.sel(x=xx,y=yy, method="nearest")
#     pwater = water_geotiffs_ds.water.sel(x=xx,y=yy, method="nearest")
#     pveg = veg_geotiffs_ds.veg.sel(x=xx,y=yy, method="nearest")
#     pdem = dem_geotiffs_ds.dem.sel(x=xx,y=yy, method="nearest")
#     pdist = dist_geotiffs_ds.dist.sel(x=xx,y=yy, method="nearest")

#     dat_wood[counter,:] = pwood.to_numpy()
#     dat_water[counter,:] = pwater.to_numpy()
#     dat_veg[counter,:] = pveg.to_numpy()
#     dat_dem[counter,:] = pdem.to_numpy()
#     dat_dist[counter,:] = pdist.to_numpy()

# np.savez('../results/LR/LR_wood/summary/LR_wood_water_veg_dem_dist_allpts_5m.npz', 
#          dat_veg=dat_veg, dat_water=dat_water, 
#          dat_wood=dat_wood, dat_dem=dat_dem, dat_dist = dat_dist,
#          x=x, y=y)

# # np.savez('../results/LR/bin_wood_water_veg_allpts.npz', dat_veg=dat_veg, dat_water=dat_water, dat_wood=dat_wood, x=x, y=y)
# # np.savez('../results/LR/probs_wood_water_veg_allpts.npz', dat_veg=dat_veg, dat_water=dat_water, dat_wood=dat_wood, x=x, y=y)

# # plt.scatter(x,y,10,np.mean(dat_wood,axis=1)); plt.show()
# # plt.scatter(x,y,10,np.mean(dat_veg,axis=1)); plt.show()
# # plt.scatter(x,y,10,np.mean(dat_water,axis=1)); plt.show()

# import pandas as pd
# for counter,time in enumerate(times):
#     d = {"dat_veg":dat_veg[:,counter], "dat_water":dat_water[:,counter], 
#             "dat_wood":dat_wood[:,counter], "dat_dem":dat_dem[:,counter], "dat_dist":dat_dist[:,counter],
#             "x":x, "y":y}
#     df = pd.DataFrame(d)
#     df.to_csv(f"../results/LR/LR_wood/summary/LR_{time}_wood_water_veg_dem_dist_allpts_5m.csv")


# #########################################################

# with np.load('../results/LR/LR_wood/summary/LR_wood_water_veg_dem_dist_allpts_5m.npz') as f:
#     dat_veg = f['dat_veg']
#     dat_water = f['dat_water']
#     dat_wood = f['dat_wood']
#     dat_x = f['x']
#     dat_y = f['y']

# dt = [datetime.strptime(time,'%Y-%m-%d') for time in times]

# plt.figure(figsize=(6,12))
# plt.subplot(311)
# plt.plot(dt,np.sum(dat_wood>0,0),'r')
# plt.subplot(312)
# plt.plot(dt,np.sum(dat_water>0,0),'b')
# plt.subplot(313)
# plt.plot(dt,np.sum(dat_veg>0,0),'g')
# plt.show()


# dem_bins = np.linspace(dat_dem.min(), 30, 50) #dat_dem.max()
# dist_bins = np.linspace(dat_dist.min(), 300, 50) #dat_dist.max()

# f, axs = plt.subplots(14, 2,figsize=(16,16))
# f.subplots_adjust(wspace=0.0, hspace=0.0)

# for counter,time in enumerate(times):
#     ind = np.where(dat_wood[:,counter]>0)[0]

#     axs[counter][0].hist(dat_dem[ind,counter], bins=dem_bins)
#     axs[counter][0].set_ylim(0,30)

#     axs[counter,0].set_ylabel(times[counter], rotation=60)

#     axs[counter][1].hist(dat_dist[ind,counter], bins=dist_bins)
#     axs[counter][1].set_ylim(0,30)

# axs[0,0].set_title('Wood binned by elevation')
# axs[0,1].set_title('Wood binned by distance to braid')

# axs[counter,0].set_xlabel('Elevation [m]')
# axs[counter,1].set_xlabel('Distance to braid centerline [m]')

# # plt.show()
# plt.savefig("../results/LR/LR_wood/summary/Wood_bin_distr_wrt_elev_distbraid.png", dpi=300, bbox_inches='tight')
# plt.close()


# #########################################################
# #########################################################

# plt.figure(figsize=(32,16))

# for k in range(len(times)):
#     tmp = dat_wood[:,k].copy()
#     xtmp = x.copy()
#     ytmp = y.copy()
#     xtmp = xtmp[tmp>0.25]
#     ytmp = ytmp[tmp>0.25]
#     tmp = tmp[tmp>0.25]

#     plt.subplot(1,14,k+1)
#     plt.scatter(xtmp,ytmp,10,tmp)
#     plt.axis('off')
#     plt.title(times[k])

# # plt.show()
# plt.savefig("../results/LR/LR_wood/summary/Wood_bin_5m_sample_history.png", dpi=300, bbox_inches='tight')
# plt.close()

# tots = []
# for k in range(len(times)):
#     tmp = dat_wood[:,k].copy()
#     xtmp = x.copy()
#     ytmp = y.copy()
#     xtmp = xtmp[tmp>0.25]
#     ytmp = ytmp[tmp>0.25]
#     tmp = tmp[tmp>0.25]
#     tots.append(100*np.sum(tmp>0)/len(x))

# plt.figure(figsize=(32,4))
# plt.plot(dt,tots,'k-',marker='o',markerfacecolor='r', markeredgecolor='w')
# plt.xlabel('Time')
# plt.ylabel('$\%$ sample points\n occupied by wood')
# # plt.show()
# plt.savefig("../results/LR/LR_wood/summary/Percent_reach_wood_bin_sample_history_5m.png", dpi=300)
# plt.close()

# #############################################################
# #########################################################
# ## correlations over time, each region of filtered wood

# def autocorr(dat):
#     result = np.correlate(dat, dat, mode='full')
#     ind = np.round(len(result)/2).astype('int')
#     return result[ind:]

# a_wood = np.zeros((len(x), len(times)-1))
# a_veg = np.zeros((len(x), len(times)-1))
# a_water = np.zeros((len(x), len(times)-1))
# for k in range(len(x)):
#     a_wood[k] = autocorr(dat_wood[k,:])
#     a_veg[k] = autocorr(dat_veg[k,:])
#     a_water[k] = autocorr(dat_water[k,:])

# plt.figure(figsize=(4,16))
# plt.subplot(131)
# plt.scatter(x,y,10,a_wood[:,0])
# plt.subplot(132)
# plt.scatter(x,y,10,a_veg[:,0])
# plt.subplot(133)
# plt.scatter(x,y,10,a_water[:,0])
# plt.show()


# s_wood = np.zeros((len(x)))
# s_veg = np.zeros((len(x)))
# s_water = np.zeros((len(x)))
# for k in range(len(x)):
#     s_wood[k] = np.std(dat_wood[k,:])
#     s_veg[k] = np.std(dat_veg[k,:])
#     s_water[k] = np.std(dat_water[k,:])

# plt.figure(figsize=(4,16))
# plt.subplot(131)
# plt.scatter(x,y,10,s_wood)
# plt.subplot(132)
# plt.scatter(x,y,10,s_veg)
# plt.subplot(133)
# plt.scatter(x,y,10,s_water)
# plt.show()








# # da = xr.tutorial.open_dataset('air_temperature')["air"].load()
# # da_2013 = da.sel(time="2013")
# # da_2014 = da.sel(time="2014")
# # da_2013["time"] = da_2013["time"].dt.strftime("%m%d%H")
# # da_2014["time"] = da_2014["time"].dt.strftime("%m%d%H")
# # da_corr = xr.corr(da_2013, da_2014, dim="time")

# # import bottleneck

# # def covariance_gufunc(x, y):
# #     return (
# #         (x - x.mean(axis=-1, keepdims=True)) * (y - y.mean(axis=-1, keepdims=True))
# #     ).mean(axis=-1)


# # def pearson_correlation_gufunc(x, y):
# #     return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))


# # def spearman_correlation_gufunc(x, y):
# #     x_ranks = bottleneck.rankdata(x, axis=-1)
# #     y_ranks = bottleneck.rankdata(y, axis=-1)
# #     return pearson_correlation_gufunc(x_ranks, y_ranks)


# # def spearman_correlation(x, y, dim):
# #     return xr.apply_ufunc(
# #         spearman_correlation_gufunc,
# #         x,
# #         y,
# #         input_core_dims=[[dim], [dim]],
# #         dask="parallelized",
# #         output_dtypes=[float],
# #     )

# # r = spearman_correlation(array1, array2, "time").compute()


# #############################################################
# #########################################################
# ## bin by elevation


# #############################################################
# #########################################################
# ## active versus non-active wood









# # plt.figure(figsize=(10,6))
# # pwood.plot(color=[.588, .294, 0.], label='wood'); pwater.plot(color='b', label='water')
# # pveg.plot(color='g', label='veg'); pdev.plot(color=[.5,.5,.5], label='dev'); 
# # plt.legend()
# # plt.xlabel('Prediction probability')
# # plt.xticks(rotation=45)
# # plt.show()


# # source_crs = wood_geotiffs_ds.rio.crs.data['init'] # Coordinate system of the file
# # target_crs = 'epsg:4326' # Global lat-lon coordinate system

# # polar_to_latlon = Transformer.from_crs(target_crs,source_crs)
# # lat, lon = polar_to_latlon.transform(x, y, direction='inverse')
# # print(lat)
# # print(lon)

# # xx=-123.60021506
# # yy=48.00648881



# ## alternative method
# # dat_wood = np.zeros((len(x),len(times)))
# # dat_water = np.zeros((len(x),len(times)))
# # dat_veg = np.zeros((len(x),len(times)))

# # for counter, (xx,yy) in tqdm(enumerate(zip(x,y))):
# #     for inner_counter, time in enumerate(times):
# #         pwood = wood_geotiffs_ds.wood.sel(x=xx,y=yy, method="nearest").sel(time=time)
# #         pwater = water_geotiffs_ds.water.sel(x=xx,y=yy, method="nearest").sel(time=time)
# #         pveg = veg_geotiffs_ds.veg.sel(x=xx,y=yy, method="nearest").sel(time=time)
# #         dat_wood[counter,inner_counter] = pwood
# #         dat_water[counter,inner_counter] = pwater
# #         dat_veg[counter,inner_counter] = pveg

# # plt.figure(figsize=(10,6))
# # pwood.plot(color=[.588, .294, 0.], label='wood'); pwater.plot(color='b', label='water')
# # pveg.plot(color='g', label='veg'); pdev.plot(color=[.5,.5,.5], label='dev'); 
# # plt.legend()
# # plt.xlabel('Prediction probability')
# # plt.xticks(rotation=45)
# # plt.show()
