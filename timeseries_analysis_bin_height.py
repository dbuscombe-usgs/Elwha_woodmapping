## Dan Buscombe, Marda Science
## Apr-June, 2023
import json, os
import rioxarray
import xarray as xr 
from glob import glob 
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client
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

# n_workers = 20
# threads_per_worker = 2
# memory_limit='50GB'
# #############################################################
# ## start client
# client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

cwd = os.getcwd()

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

# get area of each budget reach and  put in a list
A_LR = []
for g in tqdm(LRbudget_reaches):
    A_LR.append(area(g['geometry']))

A_MR = []
for g in tqdm(MRbudget_reaches):
    A_MR.append(area(g['geometry']))


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

#############################################################
# get detrended dems

# ####### MR
# MR_detrend_dem = rioxarray.open_rasterio("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_global_regrid.tif", chunks=chunksize, dtype=dtype)
# MR_detrend_dem = MR_detrend_dem.to_dataset('band').persist()
# print(MR_detrend_dem.dims)


####### MR
# MR_detrend_dem = rioxarray.open_rasterio("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_global_regrid.tif", chunks=chunksize, dtype=dtype)
# MR_detrend_dem = MR_detrend_dem.to_dataset('band').persist()
# print(MR_detrend_dem.dims)

dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_2*.tif'))
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
MR_dem_detrend_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})


# ####### LR
# LR_detrend_dem = rioxarray.open_rasterio("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/LR_DEM_detrend_global_regrid.tif", chunks=chunksize, dtype=dtype)
# LR_detrend_dem = LR_detrend_dem.to_dataset('band').persist()
# print(LR_detrend_dem.dims)


dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/LR_DEM_detrend_2*.tif'))
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
LR_dem_detrend_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})



print(MRwood_geotiffs_ds.dims)

print(LRwood_geotiffs_ds.dims)

dists = pd.read_csv('br_dists.csv')
LR = np.hstack((0,np.array(dists['LR'])))
MR = np.hstack((0,np.array(dists['MR'][:43])))


A_MR = np.array(A_MR)
A_LR = np.array(A_LR)

# S = []
# for k in [2,4,5,6,8,9,10,20,40]:
#     s = float(MR_detrend_dem[1].where(MR_detrend_dem[1]<k).sum().compute())
#     S.append(s)
#     print(s)

# LS = []
# for k in [2,4,5,6,8,9,10,20,40]:
#     s = float(LR_detrend_dem[1].where(LR_detrend_dem[1]<k).sum().compute())
#     LS.append(s)
#     print(s)

# plt.plot([2,4,5,6,8,9,10,20,40], S)
# plt.plot([2,4,5,6,8,9,10,20,40], LS)
# plt.show()


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


# #######################################################

# # sum wood pixels in each BR reach, timestamp, and 4 height bins

# MR_BR_bin1=[]
# MR_BR_bin2=[]
# MR_BR_bin3=[]
# MR_BR_bin4=[]
# MR_BR_bin5=[]
# MR_BR_bin6=[]
# MR_BR_bin7=[]
# MR_BR_bin8=[]
# MR_BR_bin9=[]
# MR_BR_bin10=[]
# MR_BR_bin11=[]
# MR_BR_bin12=[]
# MR_BR_bin13=[]
# MR_BR_bin14=[]
# for time in times:
#     tmp = MRwood_geotiffs_ds.wood.sel(time=time)
#     dem_tmp = MR_dem_detrend_geotiffs_ds.dem.sel(time=time)
#     dem_min = np.nanmin(dem_tmp)
#     if dem_min<1:
#         dem_tmp = dem_tmp-dem_min

#     bin1=[]
#     bin2=[]
#     bin3=[]
#     bin4=[]
#     bin5=[]
#     bin6=[]
#     bin7=[]
#     bin8=[]
#     bin9=[]
#     bin10=[]
#     bin11=[]
#     bin12=[]
#     bin13=[]
#     bin14=[]
#     for g in tqdm(MRbudget_reaches_redo):
#         wood_c = tmp.rio.clip([g], tmp.rio.crs).to_numpy() 
#         dem_c = dem_tmp.rio.clip([g], dem_tmp.rio.crs).to_numpy() 

#         result1 = np.nansum(wood_c*((dem_c < 1))) #(wood_c.where((dem_tmp < 5))).sum().compute().to_numpy() 
#         bin1.append(float(result1))

#         result2 = np.nansum(wood_c*((dem_c >= 1) & (dem_c < 2))) #(wood_c.where((dem_tmp >= 5) & (dem_tmp < 6))).sum().compute().to_numpy() 
#         bin2.append(float(result2))

#         result3 = np.nansum(wood_c*((dem_c >= 2) & (dem_c < 3))) #(wood_c.where((dem_tmp >= 6) & (dem_tmp < 7))).sum().compute().to_numpy() 
#         bin3.append(float(result3))

#         result4 = np.nansum(wood_c*((dem_c >= 3) & (dem_c < 4))) #(wood_c.where((dem_tmp >= 7) & (dem_tmp < 8))).sum().compute().to_numpy() 
#         bin4.append(float(result4))

#         result5 = np.nansum(wood_c*((dem_c >= 4) & (dem_c < 5))) #(wood_c.where((dem_tmp >= 8) & (dem_tmp < 9))).sum().compute().to_numpy() 
#         bin5.append(float(result5))

#         result6 = np.nansum(wood_c*((dem_c >= 5) & (dem_c < 6))) #(wood_c.where((dem_tmp >= 9) & (dem_tmp < 10))).sum().compute().to_numpy() 
#         bin6.append(float(result6))

#         result7 = np.nansum(wood_c*((dem_c >= 6) & (dem_c < 7))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin7.append(float(result7))

#         result8 = np.nansum(wood_c*((dem_c >= 7) & (dem_c < 8))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin8.append(float(result8))

#         result9 = np.nansum(wood_c*((dem_c >= 8) & (dem_c < 9))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin9.append(float(result9))

#         result10 = np.nansum(wood_c*((dem_c >= 9) & (dem_c < 10))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin10.append(float(result10))

#         result11 = np.nansum(wood_c*((dem_c >= 10) & (dem_c < 11))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin11.append(float(result11))

#         result12 = np.nansum(wood_c*((dem_c >= 11) & (dem_c < 12))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin12.append(float(result12))

#         result13 = np.nansum(wood_c*((dem_c >= 12) & (dem_c < 13))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin13.append(float(result13))        

#         result14 = np.nansum(wood_c*((dem_c > 13))) #(wood_c.where((dem_tmp > 11))).sum().compute().to_numpy() 
#         bin14.append(float(result14))        

#     MR_BR_bin1.append(bin1)
#     MR_BR_bin2.append(bin2)
#     MR_BR_bin3.append(bin3)
#     MR_BR_bin4.append(bin4)
#     MR_BR_bin5.append(bin5)
#     MR_BR_bin6.append(bin6)
#     MR_BR_bin7.append(bin7)
#     MR_BR_bin8.append(bin8)
#     MR_BR_bin9.append(bin9)
#     MR_BR_bin10.append(bin10)
#     MR_BR_bin11.append(bin11)
#     MR_BR_bin12.append(bin12)
#     MR_BR_bin13.append(bin13)
#     MR_BR_bin14.append(bin14)

#     print(MR_BR_bin3)


# #####################################################################3
# LR_BR_bin1=[]
# LR_BR_bin2=[]
# LR_BR_bin3=[]
# LR_BR_bin4=[]
# LR_BR_bin5=[]
# LR_BR_bin6=[]
# LR_BR_bin7=[]
# LR_BR_bin8=[]
# LR_BR_bin9=[]
# LR_BR_bin10=[]
# LR_BR_bin11=[]
# LR_BR_bin12=[]
# LR_BR_bin13=[]
# LR_BR_bin14=[]
# for time in times:
#     tmp = LRwood_geotiffs_ds.wood.sel(time=time)
#     dem_tmp = LR_dem_detrend_geotiffs_ds.dem.sel(time=time)
#     dem_min = np.nanmin(dem_tmp)
#     if dem_min<1:
#         dem_tmp = dem_tmp-dem_min

#     bin1=[]
#     bin2=[]
#     bin3=[]
#     bin4=[]
#     bin5=[]
#     bin6=[]
#     bin7=[]
#     bin8=[]
#     bin9=[]
#     bin10=[]
#     bin11=[]
#     bin12=[]
#     bin13=[]
#     bin14=[]
#     for g in tqdm(LRbudget_reaches_redo):
#         wood_c = tmp.rio.clip([g], tmp.rio.crs).to_numpy() 
#         dem_c = dem_tmp.rio.clip([g], dem_tmp.rio.crs).to_numpy() 

#         result1 = np.nansum(wood_c*((dem_c < 1))) #(wood_c.where((dem_tmp < 5))).sum().compute().to_numpy() 
#         bin1.append(float(result1))

#         result2 = np.nansum(wood_c*((dem_c >= 1) & (dem_c < 2))) #(wood_c.where((dem_tmp >= 5) & (dem_tmp < 6))).sum().compute().to_numpy() 
#         bin2.append(float(result2))

#         result3 = np.nansum(wood_c*((dem_c >= 2) & (dem_c < 3))) #(wood_c.where((dem_tmp >= 6) & (dem_tmp < 7))).sum().compute().to_numpy() 
#         bin3.append(float(result3))

#         result4 = np.nansum(wood_c*((dem_c >= 3) & (dem_c < 4))) #(wood_c.where((dem_tmp >= 7) & (dem_tmp < 8))).sum().compute().to_numpy() 
#         bin4.append(float(result4))

#         result5 = np.nansum(wood_c*((dem_c >= 4) & (dem_c < 5))) #(wood_c.where((dem_tmp >= 8) & (dem_tmp < 9))).sum().compute().to_numpy() 
#         bin5.append(float(result5))

#         result6 = np.nansum(wood_c*((dem_c >= 5) & (dem_c < 6))) #(wood_c.where((dem_tmp >= 9) & (dem_tmp < 10))).sum().compute().to_numpy() 
#         bin6.append(float(result6))

#         result7 = np.nansum(wood_c*((dem_c >= 6) & (dem_c < 7))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin7.append(float(result7))

#         result8 = np.nansum(wood_c*((dem_c >= 7) & (dem_c < 8))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin8.append(float(result8))

#         result9 = np.nansum(wood_c*((dem_c >= 8) & (dem_c < 9))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin9.append(float(result9))

#         result10 = np.nansum(wood_c*((dem_c >= 9) & (dem_c < 10))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin10.append(float(result10))

#         result11 = np.nansum(wood_c*((dem_c >= 10) & (dem_c < 11))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin11.append(float(result11))

#         result12 = np.nansum(wood_c*((dem_c >= 11) & (dem_c < 12))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin12.append(float(result12))

#         result13 = np.nansum(wood_c*((dem_c >= 12) & (dem_c < 13))) #(wood_c.where((dem_tmp >= 10) & (dem_tmp < 11))).sum().compute().to_numpy() 
#         bin13.append(float(result13))        

#         result14 = np.nansum(wood_c*((dem_c > 13))) #(wood_c.where((dem_tmp > 11))).sum().compute().to_numpy() 
#         bin14.append(float(result14))        

#     LR_BR_bin1.append(bin1)
#     LR_BR_bin2.append(bin2)
#     LR_BR_bin3.append(bin3)
#     LR_BR_bin4.append(bin4)
#     LR_BR_bin5.append(bin5)
#     LR_BR_bin6.append(bin6)
#     LR_BR_bin7.append(bin7)
#     LR_BR_bin8.append(bin8)
#     LR_BR_bin9.append(bin9)
#     LR_BR_bin10.append(bin10)
#     LR_BR_bin11.append(bin11)
#     LR_BR_bin12.append(bin12)
#     LR_BR_bin13.append(bin13)
#     LR_BR_bin14.append(bin14)

#     print(LR_BR_bin3)


# # ###############################################################################################

# MR_BR_bin1_scaled = np.vstack(MR_BR_bin1)/grid2sqm/A_MR
# MR_BR_bin2_scaled = np.vstack(MR_BR_bin2)/grid2sqm/A_MR
# MR_BR_bin3_scaled = np.vstack(MR_BR_bin3)/grid2sqm/A_MR
# MR_BR_bin4_scaled = np.vstack(MR_BR_bin4)/grid2sqm/A_MR
# MR_BR_bin5_scaled = np.vstack(MR_BR_bin5)/grid2sqm/A_MR
# MR_BR_bin6_scaled = np.vstack(MR_BR_bin6)/grid2sqm/A_MR
# MR_BR_bin7_scaled = np.vstack(MR_BR_bin7)/grid2sqm/A_MR
# MR_BR_bin8_scaled = np.vstack(MR_BR_bin8)/grid2sqm/A_MR
# MR_BR_bin9_scaled = np.vstack(MR_BR_bin9)/grid2sqm/A_MR
# MR_BR_bin10_scaled = np.vstack(MR_BR_bin10)/grid2sqm/A_MR
# MR_BR_bin11_scaled = np.vstack(MR_BR_bin11)/grid2sqm/A_MR
# MR_BR_bin12_scaled = np.vstack(MR_BR_bin12)/grid2sqm/A_MR
# MR_BR_bin13_scaled = np.vstack(MR_BR_bin13)/grid2sqm/A_MR
# MR_BR_bin14_scaled = np.vstack(MR_BR_bin14)/grid2sqm/A_MR


# LR_BR_bin1_scaled = np.vstack(LR_BR_bin1)/grid2sqm/A_LR
# LR_BR_bin2_scaled = np.vstack(LR_BR_bin2)/grid2sqm/A_LR
# LR_BR_bin3_scaled = np.vstack(LR_BR_bin3)/grid2sqm/A_LR
# LR_BR_bin4_scaled = np.vstack(LR_BR_bin4)/grid2sqm/A_LR
# LR_BR_bin5_scaled = np.vstack(LR_BR_bin5)/grid2sqm/A_LR
# LR_BR_bin6_scaled = np.vstack(LR_BR_bin6)/grid2sqm/A_LR
# LR_BR_bin7_scaled = np.vstack(LR_BR_bin7)/grid2sqm/A_LR
# LR_BR_bin8_scaled = np.vstack(LR_BR_bin8)/grid2sqm/A_LR
# LR_BR_bin9_scaled = np.vstack(LR_BR_bin9)/grid2sqm/A_LR
# LR_BR_bin10_scaled = np.vstack(LR_BR_bin10)/grid2sqm/A_LR
# LR_BR_bin11_scaled = np.vstack(LR_BR_bin11)/grid2sqm/A_LR
# LR_BR_bin12_scaled = np.vstack(LR_BR_bin12)/grid2sqm/A_LR
# LR_BR_bin13_scaled = np.vstack(LR_BR_bin13)/grid2sqm/A_LR
# LR_BR_bin14_scaled = np.vstack(LR_BR_bin14)/grid2sqm/A_LR

# # np.savez('summaries/Wood_time_series_bins_height_MR.npz', MR_BR_bin1_scaled = MR_BR_bin1_scaled,MR_BR_bin2_scaled = MR_BR_bin2_scaled,MR_BR_bin3_scaled = MR_BR_bin3_scaled, MR_BR_bin4_scaled = MR_BR_bin4_scaled, MR_BR_bin5_scaled = MR_BR_bin5_scaled, MR_BR_bin6_scaled=MR_BR_bin6_scaled, MR_BR_bin7_scaled = MR_BR_bin7_scaled, MR_BR_bin8_scaled=MR_BR_bin8_scaled, MR_BR_bin1=MR_BR_bin1,MR_BR_bin2=MR_BR_bin2,MR_BR_bin3=MR_BR_bin3,MR_BR_bin4=MR_BR_bin4, MR_BR_bin5=MR_BR_bin5, MR_BR_bin6=MR_BR_bin6, MR_BR_bin7=MR_BR_bin7, MR_BR_bin8=MR_BR_bin8)

# # np.savez('summaries/Wood_time_series_bins_height_LR.npz', LR_BR_bin1_scaled = LR_BR_bin1_scaled,LR_BR_bin2_scaled = LR_BR_bin2_scaled,LR_BR_bin3_scaled = LR_BR_bin3_scaled, LR_BR_bin4_scaled = LR_BR_bin4_scaled, LR_BR_bin5_scaled = LR_BR_bin5_scaled, LR_BR_bin6_scaled=LR_BR_bin6_scaled, LR_BR_bin7_scaled = LR_BR_bin7_scaled, LR_BR_bin8_scaled=LR_BR_bin8_scaled, LR_BR_bin1=LR_BR_bin1,LR_BR_bin2=LR_BR_bin2,LR_BR_bin3=LR_BR_bin3,LR_BR_bin4=LR_BR_bin4, LR_BR_bin5=LR_BR_bin5, LR_BR_bin6=LR_BR_bin6, LR_BR_bin7=LR_BR_bin7, LR_BR_bin8=LR_BR_bin8)

# np.savez('summaries/Wood_time_series_bins_height_MR_redo.npz', MR_BR_bin1_scaled = MR_BR_bin1_scaled,MR_BR_bin2_scaled = MR_BR_bin2_scaled,MR_BR_bin3_scaled = MR_BR_bin3_scaled, MR_BR_bin4_scaled = MR_BR_bin4_scaled, MR_BR_bin5_scaled = MR_BR_bin5_scaled, MR_BR_bin6_scaled=MR_BR_bin6_scaled, MR_BR_bin7_scaled = MR_BR_bin7_scaled, MR_BR_bin8_scaled=MR_BR_bin8_scaled, MR_BR_bin9_scaled=MR_BR_bin9_scaled, MR_BR_bin10_scaled=MR_BR_bin10_scaled, MR_BR_bin11_scaled=MR_BR_bin11_scaled, MR_BR_bin12_scaled=MR_BR_bin12_scaled, MR_BR_bin13_scaled=MR_BR_bin13_scaled, MR_BR_bin14_scaled=MR_BR_bin14_scaled, MR_BR_bin1=MR_BR_bin1,MR_BR_bin2=MR_BR_bin2,MR_BR_bin3=MR_BR_bin3,MR_BR_bin4=MR_BR_bin4, MR_BR_bin5=MR_BR_bin5, MR_BR_bin6=MR_BR_bin6, MR_BR_bin7=MR_BR_bin7, MR_BR_bin8=MR_BR_bin8,MR_BR_bin9=MR_BR_bin9,MR_BR_bin10=MR_BR_bin10,MR_BR_bin11=MR_BR_bin11,MR_BR_bin12=MR_BR_bin12,MR_BR_bin13=MR_BR_bin13,MR_BR_bin14=MR_BR_bin14)

# np.savez('summaries/Wood_time_series_bins_height_LR_redo.npz', LR_BR_bin1_scaled = LR_BR_bin1_scaled,LR_BR_bin2_scaled = LR_BR_bin2_scaled,LR_BR_bin3_scaled = LR_BR_bin3_scaled, LR_BR_bin4_scaled = LR_BR_bin4_scaled, LR_BR_bin5_scaled = LR_BR_bin5_scaled, LR_BR_bin6_scaled=LR_BR_bin6_scaled, LR_BR_bin7_scaled = LR_BR_bin7_scaled, LR_BR_bin8_scaled=LR_BR_bin8_scaled, LR_BR_bin9_scaled=LR_BR_bin9_scaled, LR_BR_bin10_scaled=LR_BR_bin10_scaled, LR_BR_bin11_scaled=LR_BR_bin11_scaled, LR_BR_bin12_scaled=LR_BR_bin12_scaled, LR_BR_bin13_scaled=LR_BR_bin13_scaled, LR_BR_bin14_scaled=LR_BR_bin14_scaled, LR_BR_bin1=LR_BR_bin1,LR_BR_bin2=LR_BR_bin2,LR_BR_bin3=LR_BR_bin3,LR_BR_bin4=LR_BR_bin4, LR_BR_bin5=LR_BR_bin5, LR_BR_bin6=LR_BR_bin6, LR_BR_bin7=LR_BR_bin7, LR_BR_bin8=LR_BR_bin8,LR_BR_bin9=LR_BR_bin9,LR_BR_bin10=LR_BR_bin10,LR_BR_bin11=LR_BR_bin11,LR_BR_bin12=LR_BR_bin12,LR_BR_bin13=LR_BR_bin13,LR_BR_bin14=LR_BR_bin14)



with np.load('summaries/Wood_time_series_bins_height_MR_redo.npz', allow_pickle=True) as f:
    MR_BR_bin1_scaled = f['MR_BR_bin1_scaled']
    MR_BR_bin2_scaled = f['MR_BR_bin2_scaled']
    MR_BR_bin3_scaled = f['MR_BR_bin3_scaled']
    MR_BR_bin4_scaled = f['MR_BR_bin4_scaled']
    MR_BR_bin5_scaled = f['MR_BR_bin5_scaled']
    MR_BR_bin6_scaled = f['MR_BR_bin6_scaled']
    MR_BR_bin7_scaled = f['MR_BR_bin7_scaled']
    MR_BR_bin8_scaled = f['MR_BR_bin8_scaled']
    MR_BR_bin9_scaled = f['MR_BR_bin9_scaled']
    MR_BR_bin10_scaled = f['MR_BR_bin10_scaled']
    MR_BR_bin11_scaled = f['MR_BR_bin11_scaled']
    MR_BR_bin12_scaled = f['MR_BR_bin12_scaled']
    MR_BR_bin13_scaled = f['MR_BR_bin13_scaled']
    MR_BR_bin14_scaled = f['MR_BR_bin14_scaled']

    MR_BR_bin1 = f['MR_BR_bin1']
    MR_BR_bin2 = f['MR_BR_bin2']
    MR_BR_bin3 = f['MR_BR_bin3']
    MR_BR_bin4 = f['MR_BR_bin4']
    MR_BR_bin5 = f['MR_BR_bin5']
    MR_BR_bin6 = f['MR_BR_bin6']
    MR_BR_bin7 = f['MR_BR_bin7']
    MR_BR_bin8 = f['MR_BR_bin8']
    MR_BR_bin9 = f['MR_BR_bin9']
    MR_BR_bin10 = f['MR_BR_bin10']
    MR_BR_bin11 = f['MR_BR_bin11']
    MR_BR_bin12 = f['MR_BR_bin12']
    MR_BR_bin13 = f['MR_BR_bin13']
    MR_BR_bin14 = f['MR_BR_bin14']



with np.load('summaries/Wood_time_series_bins_height_LR_redo.npz', allow_pickle=True) as f:
    LR_BR_bin1_scaled = f['LR_BR_bin1_scaled']
    LR_BR_bin2_scaled = f['LR_BR_bin2_scaled']
    LR_BR_bin3_scaled = f['LR_BR_bin3_scaled']
    LR_BR_bin4_scaled = f['LR_BR_bin4_scaled']
    LR_BR_bin5_scaled = f['LR_BR_bin5_scaled']
    LR_BR_bin6_scaled = f['LR_BR_bin6_scaled']
    LR_BR_bin7_scaled = f['LR_BR_bin7_scaled']
    LR_BR_bin8_scaled = f['LR_BR_bin8_scaled']
    LR_BR_bin9_scaled = f['LR_BR_bin9_scaled']
    LR_BR_bin10_scaled = f['LR_BR_bin10_scaled']
    LR_BR_bin11_scaled = f['LR_BR_bin11_scaled']
    LR_BR_bin12_scaled = f['LR_BR_bin12_scaled']
    LR_BR_bin13_scaled = f['LR_BR_bin13_scaled']
    LR_BR_bin14_scaled = f['LR_BR_bin14_scaled']

    LR_BR_bin4 = f['LR_BR_bin4']
    LR_BR_bin5 = f['LR_BR_bin5']
    LR_BR_bin6 = f['LR_BR_bin6']
    LR_BR_bin7 = f['LR_BR_bin7']
    LR_BR_bin8 = f['LR_BR_bin8']
    LR_BR_bin1 = f['LR_BR_bin1']
    LR_BR_bin2 = f['LR_BR_bin2']
    LR_BR_bin3 = f['LR_BR_bin3']
    LR_BR_bin9 = f['LR_BR_bin9']
    LR_BR_bin10 = f['LR_BR_bin10']
    LR_BR_bin11 = f['LR_BR_bin11']
    LR_BR_bin12 = f['LR_BR_bin12']
    LR_BR_bin13 = f['LR_BR_bin13']
    LR_BR_bin14 = f['LR_BR_bin14']

# ################ SEDIMENT

# #######################################################

# # sum sed pixels in each BR reach, timestamp, and 4 height bins

# MR_BR_bin1=[]
# MR_BR_bin2=[]
# MR_BR_bin3=[]
# MR_BR_bin4=[]
# MR_BR_bin5=[]
# MR_BR_bin6=[]
# MR_BR_bin7=[]
# MR_BR_bin8=[]
# MR_BR_bin9=[]
# MR_BR_bin10=[]
# MR_BR_bin11=[]
# MR_BR_bin12=[]
# MR_BR_bin13=[]
# MR_BR_bin14=[]
# for time in times:
#     tmp1 = MRsed_geotiffs_ds.sed.sel(time=time)
#     tmp2 = MRwood_geotiffs_ds.wood.sel(time=time)
#     dem_tmp = MR_dem_detrend_geotiffs_ds.dem.sel(time=time)
#     dem_min = np.nanmin(dem_tmp)
#     if dem_min<1:
#         dem_tmp = dem_tmp-dem_min

#     bin1=[]
#     bin2=[]
#     bin3=[]
#     bin4=[]
#     bin5=[]
#     bin6=[]
#     bin7=[]
#     bin8=[]
#     bin9=[]
#     bin10=[]
#     bin11=[]
#     bin12=[]
#     bin13=[]
#     bin14=[]
#     for g in tqdm(MRbudget_reaches_redo):
#         sed_c = tmp1.rio.clip([g], tmp1.rio.crs).to_numpy() 
#         wood_c = tmp2.rio.clip([g], tmp2.rio.crs).to_numpy() 
#         dem_c = dem_tmp.rio.clip([g], dem_tmp.rio.crs).to_numpy() 

#         result1 = np.nansum(sed_c*((dem_c < 1))) + np.nansum(wood_c*((dem_c < 1))) 
#         bin1.append(float(result1))

#         result2 = np.nansum(sed_c*((dem_c >= 1) & (dem_c < 2))) + np.nansum(wood_c*((dem_c >= 1) & (dem_c < 2))) 
#         bin2.append(float(result2))

#         result3 = np.nansum(sed_c*((dem_c >= 2) & (dem_c < 3))) + np.nansum(wood_c*((dem_c >= 2) & (dem_c < 3))) 
#         bin3.append(float(result3))

#         result4 = np.nansum(sed_c*((dem_c >= 3) & (dem_c < 4))) + np.nansum(wood_c*((dem_c >= 3) & (dem_c < 4))) 
#         bin4.append(float(result4))

#         result5 = np.nansum(sed_c*((dem_c >= 4) & (dem_c < 5)))  + np.nansum(wood_c*((dem_c >= 4) & (dem_c < 5))) 
#         bin5.append(float(result5))

#         result6 = np.nansum(sed_c*((dem_c >= 5) & (dem_c < 6))) + np.nansum(wood_c*((dem_c >= 5) & (dem_c < 6))) 
#         bin6.append(float(result6))

#         result7 = np.nansum(sed_c*((dem_c >= 6) & (dem_c < 7))) + np.nansum(wood_c*((dem_c >= 6) & (dem_c < 7))) 
#         bin7.append(float(result7))

#         result8 = np.nansum(sed_c*((dem_c >= 7) & (dem_c < 8))) + np.nansum(wood_c*((dem_c >= 7) & (dem_c < 8))) 
#         bin8.append(float(result8))

#         result9 = np.nansum(sed_c*((dem_c >= 8) & (dem_c < 9))) + np.nansum(wood_c*((dem_c >= 8) & (dem_c < 9))) 
#         bin9.append(float(result9))

#         result10 = np.nansum(sed_c*((dem_c >= 9) & (dem_c < 10))) + np.nansum(wood_c*((dem_c >= 9) & (dem_c < 10))) 
#         bin10.append(float(result10))

#         result11 = np.nansum(sed_c*((dem_c >= 10) & (dem_c < 11))) + np.nansum(wood_c*((dem_c >= 10) & (dem_c < 11))) 
#         bin11.append(float(result11))

#         result12 = np.nansum(sed_c*((dem_c >= 11) & (dem_c < 12))) + np.nansum(wood_c*((dem_c >= 11) & (dem_c < 12))) 
#         bin12.append(float(result12))

#         result13 = np.nansum(sed_c*((dem_c >= 12) & (dem_c < 13))) + np.nansum(wood_c*((dem_c >= 12) & (dem_c < 13))) 
#         bin13.append(float(result13))        

#         result14 = np.nansum(sed_c*((dem_c > 13))) + np.nansum(wood_c*((dem_c > 13))) 
#         bin14.append(float(result14))        

#     MR_BR_bin1.append(bin1)
#     MR_BR_bin2.append(bin2)
#     MR_BR_bin3.append(bin3)
#     MR_BR_bin4.append(bin4)
#     MR_BR_bin5.append(bin5)
#     MR_BR_bin6.append(bin6)
#     MR_BR_bin7.append(bin7)
#     MR_BR_bin8.append(bin8)
#     MR_BR_bin9.append(bin9)
#     MR_BR_bin10.append(bin10)
#     MR_BR_bin11.append(bin11)
#     MR_BR_bin12.append(bin12)
#     MR_BR_bin13.append(bin13)
#     MR_BR_bin14.append(bin14)

#     print(MR_BR_bin3)



# MR_BR_bin1_scaled = np.vstack(MR_BR_bin1)/grid2sqm/A_MR
# MR_BR_bin2_scaled = np.vstack(MR_BR_bin2)/grid2sqm/A_MR
# MR_BR_bin3_scaled = np.vstack(MR_BR_bin3)/grid2sqm/A_MR
# MR_BR_bin4_scaled = np.vstack(MR_BR_bin4)/grid2sqm/A_MR
# MR_BR_bin5_scaled = np.vstack(MR_BR_bin5)/grid2sqm/A_MR
# MR_BR_bin6_scaled = np.vstack(MR_BR_bin6)/grid2sqm/A_MR
# MR_BR_bin7_scaled = np.vstack(MR_BR_bin7)/grid2sqm/A_MR
# MR_BR_bin8_scaled = np.vstack(MR_BR_bin8)/grid2sqm/A_MR
# MR_BR_bin9_scaled = np.vstack(MR_BR_bin9)/grid2sqm/A_MR
# MR_BR_bin10_scaled = np.vstack(MR_BR_bin10)/grid2sqm/A_MR
# MR_BR_bin11_scaled = np.vstack(MR_BR_bin11)/grid2sqm/A_MR
# MR_BR_bin12_scaled = np.vstack(MR_BR_bin12)/grid2sqm/A_MR
# MR_BR_bin13_scaled = np.vstack(MR_BR_bin13)/grid2sqm/A_MR
# MR_BR_bin14_scaled = np.vstack(MR_BR_bin14)/grid2sqm/A_MR


# np.savez('summaries/Sed_time_series_bins_height_MR_redo.npz', MR_BR_bin1_scaled = MR_BR_bin1_scaled,MR_BR_bin2_scaled = MR_BR_bin2_scaled,MR_BR_bin3_scaled = MR_BR_bin3_scaled, MR_BR_bin4_scaled = MR_BR_bin4_scaled, MR_BR_bin5_scaled = MR_BR_bin5_scaled, MR_BR_bin6_scaled=MR_BR_bin6_scaled, MR_BR_bin7_scaled = MR_BR_bin7_scaled, MR_BR_bin8_scaled=MR_BR_bin8_scaled, MR_BR_bin9_scaled=MR_BR_bin9_scaled, MR_BR_bin10_scaled=MR_BR_bin10_scaled, MR_BR_bin11_scaled=MR_BR_bin11_scaled, MR_BR_bin12_scaled=MR_BR_bin12_scaled, MR_BR_bin13_scaled=MR_BR_bin13_scaled, MR_BR_bin14_scaled=MR_BR_bin14_scaled, MR_BR_bin1=MR_BR_bin1,MR_BR_bin2=MR_BR_bin2,MR_BR_bin3=MR_BR_bin3,MR_BR_bin4=MR_BR_bin4, MR_BR_bin5=MR_BR_bin5, MR_BR_bin6=MR_BR_bin6, MR_BR_bin7=MR_BR_bin7, MR_BR_bin8=MR_BR_bin8,MR_BR_bin9=MR_BR_bin9,MR_BR_bin10=MR_BR_bin10,MR_BR_bin11=MR_BR_bin11,MR_BR_bin12=MR_BR_bin12,MR_BR_bin13=MR_BR_bin13,MR_BR_bin14=MR_BR_bin14)



# #####################################################################3
# LR_BR_bin1=[]
# LR_BR_bin2=[]
# LR_BR_bin3=[]
# LR_BR_bin4=[]
# LR_BR_bin5=[]
# LR_BR_bin6=[]
# LR_BR_bin7=[]
# LR_BR_bin8=[]
# LR_BR_bin9=[]
# LR_BR_bin10=[]
# LR_BR_bin11=[]
# LR_BR_bin12=[]
# LR_BR_bin13=[]
# LR_BR_bin14=[]
# for time in times:
#     tmp1 = LRsed_geotiffs_ds.sed.sel(time=time)
#     tmp2 = LRwood_geotiffs_ds.wood.sel(time=time)
#     dem_tmp = LR_dem_detrend_geotiffs_ds.dem.sel(time=time)
#     dem_min = np.nanmin(dem_tmp)
#     if dem_min<1:
#         dem_tmp = dem_tmp-dem_min

#     bin1=[]
#     bin2=[]
#     bin3=[]
#     bin4=[]
#     bin5=[]
#     bin6=[]
#     bin7=[]
#     bin8=[]
#     bin9=[]
#     bin10=[]
#     bin11=[]
#     bin12=[]
#     bin13=[]
#     bin14=[]
#     for g in tqdm(LRbudget_reaches_redo):
#         sed_c = tmp1.rio.clip([g], tmp1.rio.crs).to_numpy() 
#         wood_c = tmp2.rio.clip([g], tmp2.rio.crs).to_numpy() 

#         dem_c = dem_tmp.rio.clip([g], dem_tmp.rio.crs).to_numpy() 

#         result1 = np.nansum(sed_c*((dem_c < 1))) + np.nansum(wood_c*((dem_c < 1)))
#         bin1.append(float(result1))

#         result2 = np.nansum(sed_c*((dem_c >= 1) & (dem_c < 2))) + np.nansum(wood_c*((dem_c >= 1) & (dem_c < 2))) 
#         bin2.append(float(result2))

#         result3 = np.nansum(sed_c*((dem_c >= 2) & (dem_c < 3))) + np.nansum(wood_c*((dem_c >= 2) & (dem_c < 3))) 
#         bin3.append(float(result3))

#         result4 = np.nansum(sed_c*((dem_c >= 3) & (dem_c < 4))) + np.nansum(wood_c*((dem_c >= 3) & (dem_c < 4))) 
#         bin4.append(float(result4))

#         result5 = np.nansum(sed_c*((dem_c >= 4) & (dem_c < 5))) + np.nansum(wood_c*((dem_c >= 4) & (dem_c < 5)))  
#         bin5.append(float(result5))

#         result6 = np.nansum(sed_c*((dem_c >= 5) & (dem_c < 6))) + np.nansum(wood_c*((dem_c >= 5) & (dem_c < 6)))  
#         bin6.append(float(result6))

#         result7 = np.nansum(sed_c*((dem_c >= 6) & (dem_c < 7))) + np.nansum(wood_c*((dem_c >= 6) & (dem_c < 7)))
#         bin7.append(float(result7))

#         result8 = np.nansum(sed_c*((dem_c >= 7) & (dem_c < 8))) + np.nansum(wood_c*((dem_c >= 7) & (dem_c < 8)))
#         bin8.append(float(result8))

#         result9 = np.nansum(sed_c*((dem_c >= 8) & (dem_c < 9))) + np.nansum(wood_c*((dem_c >= 8) & (dem_c < 9))) 
#         bin9.append(float(result9))

#         result10 = np.nansum(sed_c*((dem_c >= 9) & (dem_c < 10))) + np.nansum(wood_c*((dem_c >= 9) & (dem_c < 10)))
#         bin10.append(float(result10))

#         result11 = np.nansum(sed_c*((dem_c >= 10) & (dem_c < 11))) + np.nansum(wood_c*((dem_c >= 10) & (dem_c < 11))) 
#         bin11.append(float(result11))

#         result12 = np.nansum(sed_c*((dem_c >= 11) & (dem_c < 12))) + np.nansum(wood_c*((dem_c >= 11) & (dem_c < 12)))
#         bin12.append(float(result12))

#         result13 = np.nansum(sed_c*((dem_c >= 12) & (dem_c < 13))) + np.nansum(wood_c*((dem_c >= 12) & (dem_c < 13)))
#         bin13.append(float(result13))        

#         result14 = np.nansum(sed_c*((dem_c > 13))) + np.nansum(wood_c*((dem_c > 13)))
#         bin14.append(float(result14))        

#     LR_BR_bin1.append(bin1)
#     LR_BR_bin2.append(bin2)
#     LR_BR_bin3.append(bin3)
#     LR_BR_bin4.append(bin4)
#     LR_BR_bin5.append(bin5)
#     LR_BR_bin6.append(bin6)
#     LR_BR_bin7.append(bin7)
#     LR_BR_bin8.append(bin8)
#     LR_BR_bin9.append(bin9)
#     LR_BR_bin10.append(bin10)
#     LR_BR_bin11.append(bin11)
#     LR_BR_bin12.append(bin12)
#     LR_BR_bin13.append(bin13)
#     LR_BR_bin14.append(bin14)

#     print(LR_BR_bin3)

# ###############################################################################################

# LR_BR_bin1_scaled = np.vstack(LR_BR_bin1)/grid2sqm/A_LR
# LR_BR_bin2_scaled = np.vstack(LR_BR_bin2)/grid2sqm/A_LR
# LR_BR_bin3_scaled = np.vstack(LR_BR_bin3)/grid2sqm/A_LR
# LR_BR_bin4_scaled = np.vstack(LR_BR_bin4)/grid2sqm/A_LR
# LR_BR_bin5_scaled = np.vstack(LR_BR_bin5)/grid2sqm/A_LR
# LR_BR_bin6_scaled = np.vstack(LR_BR_bin6)/grid2sqm/A_LR
# LR_BR_bin7_scaled = np.vstack(LR_BR_bin7)/grid2sqm/A_LR
# LR_BR_bin8_scaled = np.vstack(LR_BR_bin8)/grid2sqm/A_LR
# LR_BR_bin9_scaled = np.vstack(LR_BR_bin9)/grid2sqm/A_LR
# LR_BR_bin10_scaled = np.vstack(LR_BR_bin10)/grid2sqm/A_LR
# LR_BR_bin11_scaled = np.vstack(LR_BR_bin11)/grid2sqm/A_LR
# LR_BR_bin12_scaled = np.vstack(LR_BR_bin12)/grid2sqm/A_LR
# LR_BR_bin13_scaled = np.vstack(LR_BR_bin13)/grid2sqm/A_LR
# LR_BR_bin14_scaled = np.vstack(LR_BR_bin14)/grid2sqm/A_LR


# np.savez('summaries/Sed_time_series_bins_height_LR_redo.npz', LR_BR_bin1_scaled = LR_BR_bin1_scaled,LR_BR_bin2_scaled = LR_BR_bin2_scaled,LR_BR_bin3_scaled = LR_BR_bin3_scaled, LR_BR_bin4_scaled = LR_BR_bin4_scaled, LR_BR_bin5_scaled = LR_BR_bin5_scaled, LR_BR_bin6_scaled=LR_BR_bin6_scaled, LR_BR_bin7_scaled = LR_BR_bin7_scaled, LR_BR_bin8_scaled=LR_BR_bin8_scaled, LR_BR_bin9_scaled=LR_BR_bin9_scaled, LR_BR_bin10_scaled=LR_BR_bin10_scaled, LR_BR_bin11_scaled=LR_BR_bin11_scaled, LR_BR_bin12_scaled=LR_BR_bin12_scaled, LR_BR_bin13_scaled=LR_BR_bin13_scaled, LR_BR_bin14_scaled=LR_BR_bin14_scaled, LR_BR_bin1=LR_BR_bin1,LR_BR_bin2=LR_BR_bin2,LR_BR_bin3=LR_BR_bin3,LR_BR_bin4=LR_BR_bin4, LR_BR_bin5=LR_BR_bin5, LR_BR_bin6=LR_BR_bin6, LR_BR_bin7=LR_BR_bin7, LR_BR_bin8=LR_BR_bin8,LR_BR_bin9=LR_BR_bin9,LR_BR_bin10=LR_BR_bin10,LR_BR_bin11=LR_BR_bin11,LR_BR_bin12=LR_BR_bin12,LR_BR_bin13=LR_BR_bin13,LR_BR_bin14=LR_BR_bin14)





with np.load('summaries/Sed_time_series_bins_height_MR_redo.npz', allow_pickle=True) as f:
    sMR_BR_bin1_scaled = f['MR_BR_bin1_scaled']
    sMR_BR_bin2_scaled = f['MR_BR_bin2_scaled']
    sMR_BR_bin3_scaled = f['MR_BR_bin3_scaled']
    sMR_BR_bin4_scaled = f['MR_BR_bin4_scaled']
    sMR_BR_bin5_scaled = f['MR_BR_bin5_scaled']
    sMR_BR_bin6_scaled = f['MR_BR_bin6_scaled']
    sMR_BR_bin7_scaled = f['MR_BR_bin7_scaled']
    sMR_BR_bin8_scaled = f['MR_BR_bin8_scaled']
    sMR_BR_bin9_scaled = f['MR_BR_bin9_scaled']
    sMR_BR_bin10_scaled = f['MR_BR_bin10_scaled']
    sMR_BR_bin11_scaled = f['MR_BR_bin11_scaled']
    sMR_BR_bin12_scaled = f['MR_BR_bin12_scaled']
    sMR_BR_bin13_scaled = f['MR_BR_bin13_scaled']
    sMR_BR_bin14_scaled = f['MR_BR_bin14_scaled']

    sMR_BR_bin1 = f['MR_BR_bin1']
    sMR_BR_bin2 = f['MR_BR_bin2']
    sMR_BR_bin3 = f['MR_BR_bin3']
    sMR_BR_bin4 = f['MR_BR_bin4']
    sMR_BR_bin5 = f['MR_BR_bin5']
    sMR_BR_bin6 = f['MR_BR_bin6']
    sMR_BR_bin7 = f['MR_BR_bin7']
    sMR_BR_bin8 = f['MR_BR_bin8']
    sMR_BR_bin9 = f['MR_BR_bin9']
    sMR_BR_bin10 = f['MR_BR_bin10']
    sMR_BR_bin11 = f['MR_BR_bin11']
    sMR_BR_bin12 = f['MR_BR_bin12']
    sMR_BR_bin13 = f['MR_BR_bin13']
    sMR_BR_bin14 = f['MR_BR_bin14']



with np.load('summaries/Sed_time_series_bins_height_LR_redo.npz', allow_pickle=True) as f:
    sLR_BR_bin1_scaled = f['LR_BR_bin1_scaled']
    sLR_BR_bin2_scaled = f['LR_BR_bin2_scaled']
    sLR_BR_bin3_scaled = f['LR_BR_bin3_scaled']
    sLR_BR_bin4_scaled = f['LR_BR_bin4_scaled']
    sLR_BR_bin5_scaled = f['LR_BR_bin5_scaled']
    sLR_BR_bin6_scaled = f['LR_BR_bin6_scaled']
    sLR_BR_bin7_scaled = f['LR_BR_bin7_scaled']
    sLR_BR_bin8_scaled = f['LR_BR_bin8_scaled']
    sLR_BR_bin9_scaled = f['LR_BR_bin9_scaled']
    sLR_BR_bin10_scaled = f['LR_BR_bin10_scaled']
    sLR_BR_bin11_scaled = f['LR_BR_bin11_scaled']
    sLR_BR_bin12_scaled = f['LR_BR_bin12_scaled']
    sLR_BR_bin13_scaled = f['LR_BR_bin13_scaled']
    sLR_BR_bin14_scaled = f['LR_BR_bin14_scaled']

    sLR_BR_bin1 = f['LR_BR_bin1']
    sLR_BR_bin2 = f['LR_BR_bin2']
    sLR_BR_bin3 = f['LR_BR_bin3']
    sLR_BR_bin4 = f['LR_BR_bin4']
    sLR_BR_bin5 = f['LR_BR_bin5']
    sLR_BR_bin6 = f['LR_BR_bin6']
    sLR_BR_bin7 = f['LR_BR_bin7']
    sLR_BR_bin8 = f['LR_BR_bin8']
    sLR_BR_bin9 = f['LR_BR_bin9']
    sLR_BR_bin10 = f['LR_BR_bin10']
    sLR_BR_bin11 = f['LR_BR_bin11']
    sLR_BR_bin12 = f['LR_BR_bin12']
    sLR_BR_bin13 = f['LR_BR_bin13']
    sLR_BR_bin14 = f['LR_BR_bin14']



sLR_BR_bin1 = sLR_BR_bin1 + LR_BR_bin1
sLR_BR_bin2 = sLR_BR_bin2 + LR_BR_bin2 
sLR_BR_bin3 = sLR_BR_bin3 + LR_BR_bin3  
sLR_BR_bin4 = sLR_BR_bin4 + LR_BR_bin4 
sLR_BR_bin5 = sLR_BR_bin5 + LR_BR_bin5  
sLR_BR_bin6 = sLR_BR_bin6 + LR_BR_bin6  
sLR_BR_bin7 = sLR_BR_bin7 + LR_BR_bin7  
sLR_BR_bin8 = sLR_BR_bin8 + LR_BR_bin8  
sLR_BR_bin9 = sLR_BR_bin9 + LR_BR_bin9  
sLR_BR_bin10 = sLR_BR_bin10 + LR_BR_bin10  
sLR_BR_bin11 = sLR_BR_bin11 + LR_BR_bin11  
sLR_BR_bin12 = sLR_BR_bin12 + LR_BR_bin12  
sLR_BR_bin13 = sLR_BR_bin13 + LR_BR_bin13
sLR_BR_bin14 = sLR_BR_bin14 + LR_BR_bin14



sMR_BR_bin1 = sMR_BR_bin1 + MR_BR_bin1
sMR_BR_bin2 = sMR_BR_bin2 + MR_BR_bin2 
sMR_BR_bin3 = sMR_BR_bin3 + MR_BR_bin3  
sMR_BR_bin4 = sMR_BR_bin4 + MR_BR_bin4 
sMR_BR_bin5 = sMR_BR_bin5 + MR_BR_bin5  
sMR_BR_bin6 = sMR_BR_bin6 + MR_BR_bin6  
sMR_BR_bin7 = sMR_BR_bin7 + MR_BR_bin7  
sMR_BR_bin8 = sMR_BR_bin8 + MR_BR_bin8  
sMR_BR_bin9 = sMR_BR_bin9 + MR_BR_bin9  
sMR_BR_bin10 = sMR_BR_bin10 + MR_BR_bin10  
sMR_BR_bin11 = sMR_BR_bin11 + MR_BR_bin11  
sMR_BR_bin12 = sMR_BR_bin12 + MR_BR_bin12  
sMR_BR_bin13 = sMR_BR_bin13 + MR_BR_bin13
sMR_BR_bin14 = sMR_BR_bin14 + MR_BR_bin14

hght = np.arange(0.5,14.5,1)

########################################
plt.figure(figsize=(14,14))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

################
## MR
plt.subplot(421)
im1wood = np.vstack((np.mean(MR_BR_bin1,axis=1),np.mean(MR_BR_bin2,axis=1),np.mean(MR_BR_bin3,axis=1),np.mean(MR_BR_bin4,axis=1),np.mean(MR_BR_bin5,axis=1),np.mean(MR_BR_bin6,axis=1),np.mean(MR_BR_bin7,axis=1),np.mean(MR_BR_bin8,axis=1),np.mean(MR_BR_bin9,axis=1),np.mean(MR_BR_bin10,axis=1),np.mean(MR_BR_bin11,axis=1),np.mean(MR_BR_bin12,axis=1),np.mean(MR_BR_bin13,axis=1),np.mean(MR_BR_bin14,axis=1)))
plt.imshow(np.flipud(im1wood), cmap='inferno', extent=[dt[1], dt[-1],1,14], aspect='auto')
plt.ylabel("Height (m)")
cb=plt.colorbar(); cb.set_label(r"Reach-averaged wood area (m$^2$)")
plt.title('a) MR', loc='left'); 

plt.subplot(423)
im1sed = np.vstack((np.mean(sMR_BR_bin1,axis=1),np.mean(sMR_BR_bin2,axis=1),np.mean(sMR_BR_bin3,axis=1),np.mean(sMR_BR_bin4,axis=1),np.mean(sMR_BR_bin5,axis=1),np.mean(sMR_BR_bin6,axis=1),np.mean(sMR_BR_bin7,axis=1),np.mean(sMR_BR_bin8,axis=1),np.mean(sMR_BR_bin9,axis=1),np.mean(sMR_BR_bin10,axis=1),np.mean(sMR_BR_bin11,axis=1),np.mean(sMR_BR_bin12,axis=1),np.mean(sMR_BR_bin13,axis=1),np.mean(sMR_BR_bin14,axis=1)))
plt.imshow(np.flipud(im1sed), cmap='inferno', extent=[dt[1], dt[-1],1,14], aspect='auto')
plt.ylabel("Height (m)")
cb=plt.colorbar(); cb.set_label(r"Reach-averaged sediment area (m$^2$)")
plt.title('c) MR', loc='left'); 

plt.subplot(425)
plt.semilogx(np.mean(im1wood,axis=1),hght, 'k', label='wood')
plt.plot(np.mean(im1sed,axis=1),hght, 'r--', label='sediment')
plt.legend()
plt.ylabel("Height (m)"); plt.xlabel(r"Reach-and time-averaged area (m$^2$)")
plt.title('e) MR', loc='left'); 

plt.subplot(427)
plt.plot( (np.mean(im1wood,axis=1)/np.mean(im1wood,axis=1).max()) / (np.mean(im1sed,axis=1)/np.mean(im1sed,axis=1).max()) ,hght, 'k')
plt.ylabel("Height (m)"); plt.xlabel(r"Normalized wood: normalized sediment ratio (-)")
plt.title('g) MR', loc='left'); 
plt.axvline(1.0,color='b', linestyle=':', lw=2)

################
## LR
plt.subplot(422)
im2wood = np.vstack((np.mean(LR_BR_bin1,axis=1),np.mean(LR_BR_bin2,axis=1),np.mean(LR_BR_bin3,axis=1),np.mean(LR_BR_bin4,axis=1),np.mean(LR_BR_bin5,axis=1),np.mean(LR_BR_bin6,axis=1),np.mean(LR_BR_bin7,axis=1),np.mean(LR_BR_bin8,axis=1),np.mean(LR_BR_bin9,axis=1),np.mean(LR_BR_bin10,axis=1),np.mean(LR_BR_bin11,axis=1),np.mean(LR_BR_bin12,axis=1),np.mean(LR_BR_bin13,axis=1),np.mean(LR_BR_bin14,axis=1)))
plt.imshow(np.flipud(im2wood), cmap='inferno', extent=[dt[1], dt[-1],1,14], aspect='auto')
plt.ylabel("Height (m)")
cb=plt.colorbar(); cb.set_label(r"Reach-averaged wood area (m$^2$)")
plt.title('b) LR', loc='left'); 

plt.subplot(424)
im2sed = np.vstack((np.mean(sLR_BR_bin1,axis=1),np.mean(sLR_BR_bin2,axis=1),np.mean(sLR_BR_bin3,axis=1),np.mean(sLR_BR_bin4,axis=1),np.mean(sLR_BR_bin5,axis=1),np.mean(sLR_BR_bin6,axis=1),np.mean(sLR_BR_bin7,axis=1),np.mean(sLR_BR_bin8,axis=1),np.mean(sLR_BR_bin9,axis=1),np.mean(sLR_BR_bin10,axis=1),np.mean(sLR_BR_bin11,axis=1),np.mean(sLR_BR_bin12,axis=1),np.mean(sLR_BR_bin13,axis=1),np.mean(sLR_BR_bin14,axis=1)))
plt.imshow(np.flipud(im2sed), cmap='inferno', extent=[dt[1], dt[-1],1,14], aspect='auto')
plt.ylabel("Height (m)")
cb=plt.colorbar(); cb.set_label(r"Reach-averaged sediment area (m$^2$)")
plt.title('d) LR', loc='left'); 

plt.subplot(426)
plt.semilogx(np.mean(im2wood,axis=1),hght, 'k', label='wood')
plt.plot(np.mean(im2sed,axis=1),hght, 'r--', label='sediment')
plt.legend()
plt.ylabel("Height (m)"); plt.xlabel(r"Reach-and time-averaged area (m$^2$)")
plt.title('f) LR', loc='left'); 

plt.subplot(428)
plt.plot( (np.mean(im2wood,axis=1)/np.mean(im2wood,axis=1).max()) / (np.mean(im2sed,axis=1)/np.mean(im2sed,axis=1).max()) ,hght, 'k')
plt.ylabel("Height (m)"); plt.xlabel(r"Normalized wood: normalized sediment ratio (-)")
plt.title('h) LR', loc='left'); 
plt.axvline(1.0,color='b', linestyle=':', lw=2)


plt.savefig("summaries/MR_LR_wood_sediment_average_binned_height.png", dpi=300, bbox_inches="tight")

plt.close()



### number - area relationship per height


### make this a box and whiskers plot 



# ########################################
# plt.figure(figsize=(16,6))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(231)
# y = [np.mean(np.vstack(MR_BR_bin1)/grid2sqm),np.mean(np.vstack(MR_BR_bin2)/grid2sqm),np.mean(np.vstack(MR_BR_bin3)/grid2sqm),np.mean(np.vstack(MR_BR_bin4)/grid2sqm), np.mean(np.vstack(MR_BR_bin5)/grid2sqm), np.mean(np.vstack(MR_BR_bin6)/grid2sqm), np.mean(np.vstack(MR_BR_bin7)/grid2sqm), np.mean(np.vstack(MR_BR_bin8)/grid2sqm),np.mean(np.vstack(MR_BR_bin9)/grid2sqm),np.mean(np.vstack(MR_BR_bin10)/grid2sqm),np.mean(np.vstack(MR_BR_bin11)/grid2sqm),np.mean(np.vstack(MR_BR_bin12)/grid2sqm),np.mean(np.vstack(MR_BR_bin13)/grid2sqm),np.mean(np.vstack(MR_BR_bin14)/grid2sqm)]

# y2 = [np.std(np.vstack(MR_BR_bin1)/grid2sqm),np.std(np.vstack(MR_BR_bin2)/grid2sqm),np.std(np.vstack(MR_BR_bin3)/grid2sqm),np.std(np.vstack(MR_BR_bin4)/grid2sqm), np.std(np.vstack(MR_BR_bin5)/grid2sqm), np.std(np.vstack(MR_BR_bin6)/grid2sqm), np.std(np.vstack(MR_BR_bin7)/grid2sqm), np.std(np.vstack(MR_BR_bin8)/grid2sqm),np.std(np.vstack(MR_BR_bin9)/grid2sqm),np.std(np.vstack(MR_BR_bin10)/grid2sqm),np.std(np.vstack(MR_BR_bin11)/grid2sqm),np.std(np.vstack(MR_BR_bin12)/grid2sqm),np.std(np.vstack(MR_BR_bin13)/grid2sqm),np.std(np.vstack(MR_BR_bin14)/grid2sqm)]

# x = np.arange(0.5,14.5,1)
# # x=np.array([3.5,4.5,5.5,6.5,7.5,8.5,9.5,11])-3
# # x=np.array([6.5,7.5,8.5,9.5,11])-3
# plt.semilogx(y,x,'k-o', label='Mean')
# plt.plot(y2,x,'r-s',label='Stdev.')
# # plt.plot(np.array(y2)/np.array(y),x,'b-x',label='CoV')
# plt.xlabel(r'Wood area (m$^2$)')
# plt.ylabel(r'Height above river (m)')
# plt.legend(fontsize=7)
# plt.title('a) MR', loc='left')

# plt.subplot(232)
# plt.semilogy(dt, np.cumsum(np.mean(MR_BR_bin1,axis=1))+np.cumsum(np.mean(MR_BR_bin2,axis=1))+np.cumsum(np.mean(MR_BR_bin3,axis=1))+np.cumsum(np.mean(MR_BR_bin4,axis=1)),  label='h <3.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin5,axis=1)), label='3.5m<= h <4.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin6,axis=1)), label='4.5m<= h <5.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin7,axis=1)), label='5.5m<= h <6.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin8,axis=1)), label='6.5m<= h <7.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin8,axis=1)), label='7.5m<= h <8.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin10,axis=1)), label='9.5m<= h <9.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin11,axis=1)), label='9.5m<= h <10.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin12,axis=1)), label='10.5m<= h <11.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin13,axis=1))+np.cumsum(np.mean(MR_BR_bin14,axis=1)), label='h>11.5m')
# # plt.plot(dt, np.cumsum(np.mean(MR_BR_bin4,axis=1)+np.mean(MR_BR_bin5,axis=1)+np.mean(MR_BR_bin6,axis=1)+np.mean(MR_BR_bin7,axis=1)+np.mean(MR_BR_bin8,axis=1)), 'k', lw=2, label='all bins')
# plt.legend(fontsize=7)
# plt.ylabel(r'Cumulative sum of wood (m$^2$)')
# plt.title('b) MR', loc='left')

# plt.subplot(233)
# # plt.semilogy(MR, np.cumsum(np.mean(MR_BR_bin1,axis=0))+np.cumsum(np.mean(MR_BR_bin2,axis=0))+np.cumsum(np.mean(MR_BR_bin3,axis=0))+np.cumsum(np.mean(MR_BR_bin4,axis=0)), label='h <3.5m')
# # plt.plot(MR, np.cumsum(np.mean(MR_BR_bin5,axis=0)),  label='3.5m<= h <4.5m')
# # plt.plot(MR, np.cumsum(np.mean(MR_BR_bin6,axis=0)),label='4.5m<= h <5.5m')
# # plt.plot(MR, np.cumsum(np.mean(MR_BR_bin7,axis=0)), label='5.5m<= h <6.5m')
# # plt.plot(MR, np.cumsum(np.mean(MR_BR_bin8,axis=0)), label='6.5m<= h <8m')
# # plt.plot(MR, np.cumsum(np.mean(MR_BR_bin4,axis=0)+np.mean(MR_BR_bin5,axis=0)+np.mean(MR_BR_bin6,axis=0)+np.mean(MR_BR_bin7,axis=0)+np.mean(MR_BR_bin8,axis=0)), 'k', lw=2, label='all bins')
# plt.semilogy(MR, np.cumsum(np.mean(MR_BR_bin1,axis=0))+np.cumsum(np.mean(MR_BR_bin2,axis=0))+np.cumsum(np.mean(MR_BR_bin3,axis=0))+np.cumsum(np.mean(MR_BR_bin4,axis=0)),  label='h <3.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin5,axis=0)), label='3.5m<= h <4.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin6,axis=0)), label='4.5m<= h <5.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin7,axis=0)), label='5.5m<= h <6.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin8,axis=0)), label='6.5m<= h <7.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin8,axis=0)), label='7.5m<= h <8.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin10,axis=0)), label='9.5m<= h <9.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin11,axis=0)), label='9.5m<= h <10.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin12,axis=0)), label='10.5m<= h <11.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin13,axis=0))+np.cumsum(np.mean(MR_BR_bin14,axis=0)), label='h>11.5m')

# plt.legend(fontsize=7)
# plt.ylabel(r'Cumulative sum of wood (m$^2$)')
# plt.xlabel("Distance downstream (km)")
# plt.title('c) MR', loc='left')


# plt.subplot(234)
# y = [np.mean(np.vstack(LR_BR_bin1)/grid2sqm),np.mean(np.vstack(LR_BR_bin2)/grid2sqm),np.mean(np.vstack(LR_BR_bin3)/grid2sqm),np.mean(np.vstack(LR_BR_bin4)/grid2sqm), np.mean(np.vstack(LR_BR_bin5)/grid2sqm), np.mean(np.vstack(LR_BR_bin6)/grid2sqm), np.mean(np.vstack(LR_BR_bin7)/grid2sqm), np.mean(np.vstack(LR_BR_bin8)/grid2sqm),np.mean(np.vstack(LR_BR_bin9)/grid2sqm),np.mean(np.vstack(LR_BR_bin10)/grid2sqm),np.mean(np.vstack(LR_BR_bin11)/grid2sqm),np.mean(np.vstack(LR_BR_bin12)/grid2sqm),np.mean(np.vstack(LR_BR_bin13)/grid2sqm),np.mean(np.vstack(LR_BR_bin14)/grid2sqm)]

# y2 = [np.std(np.vstack(LR_BR_bin1)/grid2sqm),np.std(np.vstack(LR_BR_bin2)/grid2sqm),np.std(np.vstack(LR_BR_bin3)/grid2sqm),np.std(np.vstack(LR_BR_bin4)/grid2sqm), np.std(np.vstack(LR_BR_bin5)/grid2sqm), np.std(np.vstack(LR_BR_bin6)/grid2sqm), np.std(np.vstack(LR_BR_bin7)/grid2sqm), np.std(np.vstack(LR_BR_bin8)/grid2sqm),np.std(np.vstack(LR_BR_bin9)/grid2sqm),np.std(np.vstack(LR_BR_bin10)/grid2sqm),np.std(np.vstack(LR_BR_bin11)/grid2sqm),np.std(np.vstack(LR_BR_bin12)/grid2sqm),np.std(np.vstack(LR_BR_bin13)/grid2sqm),np.std(np.vstack(LR_BR_bin14)/grid2sqm)]

# plt.semilogx(y,x,'k-o', label='Mean')
# plt.plot(y2,x,'r-s',label='Stdev.')
# # plt.plot(np.array(y2)/np.array(y),x,'b-x',label='CoV')
# plt.xlabel(r'Wood area (m$^2$)')
# plt.ylabel(r'Height above river (m)')
# plt.legend(fontsize=7)
# plt.title('d) LR', loc='left')


# plt.subplot(235)
# plt.semilogy(dt, np.cumsum(np.mean(LR_BR_bin1,axis=1))+np.cumsum(np.mean(LR_BR_bin2,axis=1))+np.cumsum(np.mean(LR_BR_bin3,axis=1))+np.cumsum(np.mean(LR_BR_bin4,axis=1)),  label='h <3.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin5,axis=1)), label='3.5m<= h <4.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin6,axis=1)), label='4.5m<= h <5.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin7,axis=1)), label='5.5m<= h <6.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin8,axis=1)), label='6.5m<= h <7.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin8,axis=1)), label='7.5m<= h <8.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin10,axis=1)), label='9.5m<= h <9.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin11,axis=1)), label='9.5m<= h <10.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin12,axis=1)), label='10.5m<= h <11.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin13,axis=1))+np.cumsum(np.mean(LR_BR_bin14,axis=1)), label='h>11.5m')
# plt.title('e) LR', loc='left')
# plt.ylabel(r'Cumulative sum of wood (m$^2$)')

# plt.subplot(236)
# plt.semilogy(LR, np.cumsum(np.mean(LR_BR_bin1,axis=0))+np.cumsum(np.mean(LR_BR_bin2,axis=0))+np.cumsum(np.mean(LR_BR_bin3,axis=0))+np.cumsum(np.mean(LR_BR_bin4,axis=0)),  label='h <3.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin5,axis=0)), label='3.5m<= h <4.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin6,axis=0)), label='4.5m<= h <5.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin7,axis=0)), label='5.5m<= h <6.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin8,axis=0)), label='6.5m<= h <7.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin8,axis=0)), label='7.5m<= h <8.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin10,axis=0)), label='9.5m<= h <9.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin11,axis=0)), label='9.5m<= h <10.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin12,axis=0)), label='10.5m<= h <11.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin13,axis=0))+np.cumsum(np.mean(LR_BR_bin14,axis=0)), label='h>11.5m')

# # plt.plot(LR, np.cumsum(np.mean(LR_BR_bin4,axis=0)+np.mean(LR_BR_bin5,axis=0)+np.mean(LR_BR_bin6,axis=0)+np.mean(LR_BR_bin7,axis=0)+np.mean(LR_BR_bin8,axis=0)), 'k', lw=2, label='all bins')
# plt.legend(fontsize=7)
# plt.ylabel(r'Cumulative sum of wood (m$^2$)')
# plt.xlabel("Distance downstream (km)")
# plt.title('f) LR', loc='left')

# # plt.show()
# # plt.savefig("LR_wood_spacetime_plots_binned_height.png", dpi=300, bbox_inches="tight")
# plt.savefig("summaries/MR_LR_wood_spacetime_plots_binned_height.png", dpi=300, bbox_inches="tight")

# plt.close()

####################################################



# ########################################
# plt.figure(figsize=(14,7))
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

# plt.subplot(231)
# im1 = np.vstack((np.mean(MR_BR_bin1,axis=1),np.mean(MR_BR_bin2,axis=1),np.mean(MR_BR_bin3,axis=1),np.mean(MR_BR_bin4,axis=1),np.mean(MR_BR_bin5,axis=1),np.mean(MR_BR_bin6,axis=1),np.mean(MR_BR_bin7,axis=1),np.mean(MR_BR_bin8,axis=1),np.mean(MR_BR_bin9,axis=1),np.mean(MR_BR_bin10,axis=1),np.mean(MR_BR_bin11,axis=1),np.mean(MR_BR_bin12,axis=1),np.mean(MR_BR_bin13,axis=1),np.mean(MR_BR_bin14,axis=1)))

# plt.imshow(np.flipud(im1), cmap='inferno', extent=[dt[1], dt[-1],1,14], aspect='auto')
# plt.ylabel("Height (m)")
# cb=plt.colorbar(); cb.set_label(r"Mean sediment area (m$^2$)")
# plt.title('a) MR', loc='left'); 

# plt.subplot(232)
# im2 = np.vstack((np.mean(LR_BR_bin1,axis=1),np.mean(LR_BR_bin2,axis=1),np.mean(LR_BR_bin3,axis=1),np.mean(LR_BR_bin4,axis=1),np.mean(LR_BR_bin5,axis=1),np.mean(LR_BR_bin6,axis=1),np.mean(LR_BR_bin7,axis=1),np.mean(LR_BR_bin8,axis=1),np.mean(LR_BR_bin9,axis=1),np.mean(LR_BR_bin10,axis=1),np.mean(LR_BR_bin11,axis=1),np.mean(LR_BR_bin12,axis=1),np.mean(LR_BR_bin13,axis=1),np.mean(LR_BR_bin14,axis=1)))

# plt.imshow(np.flipud(im2), cmap='inferno', extent=[dt[1], dt[-1],1,14], aspect='auto')
# plt.ylabel("Height (m)")
# cb=plt.colorbar(); cb.set_label(r"Mean sediment area (m$^2$)")
# plt.title('b) LR', loc='left'); 

# hght = np.arange(0.5,14.5,1)
# plt.subplot(233)
# plt.plot(np.mean(im1,axis=1),hght, 'k', label='MR')
# plt.plot(np.mean(im2,axis=1),hght, 'r--',label='LR')
# # plt.errorbar(np.mean(im1,axis=1),hght,np.std(im1,axis=1),1, color='k', label='MR')
# # plt.errorbar(np.mean(im2,axis=1),hght,np.std(im2,axis=1),1, color='r', linestyle='--',label='LR')

# plt.legend()
# plt.ylabel("Height (m)"); plt.xlabel(r"Mean sediment area (m$^2$)")
# plt.title('c) MR, LR', loc='left'); 

# # plt.savefig("LR_wood_spacetime_plots_binned_height.png", dpi=300, bbox_inches="tight")
# plt.savefig("summaries/MR_LR_sed_average_binned_height.png", dpi=300, bbox_inches="tight")

# plt.close()


### number - area relationship per height


### make this a box and whiskers plot 



# ########################################
# plt.figure(figsize=(16,6))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(231)
# y = [np.mean(np.vstack(MR_BR_bin1)/grid2sqm),np.mean(np.vstack(MR_BR_bin2)/grid2sqm),np.mean(np.vstack(MR_BR_bin3)/grid2sqm),np.mean(np.vstack(MR_BR_bin4)/grid2sqm), np.mean(np.vstack(MR_BR_bin5)/grid2sqm), np.mean(np.vstack(MR_BR_bin6)/grid2sqm), np.mean(np.vstack(MR_BR_bin7)/grid2sqm), np.mean(np.vstack(MR_BR_bin8)/grid2sqm),np.mean(np.vstack(MR_BR_bin9)/grid2sqm),np.mean(np.vstack(MR_BR_bin10)/grid2sqm),np.mean(np.vstack(MR_BR_bin11)/grid2sqm),np.mean(np.vstack(MR_BR_bin12)/grid2sqm),np.mean(np.vstack(MR_BR_bin13)/grid2sqm),np.mean(np.vstack(MR_BR_bin14)/grid2sqm)]

# y2 = [np.std(np.vstack(MR_BR_bin1)/grid2sqm),np.std(np.vstack(MR_BR_bin2)/grid2sqm),np.std(np.vstack(MR_BR_bin3)/grid2sqm),np.std(np.vstack(MR_BR_bin4)/grid2sqm), np.std(np.vstack(MR_BR_bin5)/grid2sqm), np.std(np.vstack(MR_BR_bin6)/grid2sqm), np.std(np.vstack(MR_BR_bin7)/grid2sqm), np.std(np.vstack(MR_BR_bin8)/grid2sqm),np.std(np.vstack(MR_BR_bin9)/grid2sqm),np.std(np.vstack(MR_BR_bin10)/grid2sqm),np.std(np.vstack(MR_BR_bin11)/grid2sqm),np.std(np.vstack(MR_BR_bin12)/grid2sqm),np.std(np.vstack(MR_BR_bin13)/grid2sqm),np.std(np.vstack(MR_BR_bin14)/grid2sqm)]

# x = np.arange(0.5,14.5,1)
# # x=np.array([3.5,4.5,5.5,6.5,7.5,8.5,9.5,11])-3
# # x=np.array([6.5,7.5,8.5,9.5,11])-3
# plt.semilogx(y,x,'k-o', label='Mean')
# plt.plot(y2,x,'r-s',label='Stdev.')
# # plt.plot(np.array(y2)/np.array(y),x,'b-x',label='CoV')
# plt.xlabel(r'Sediment area (m$^2$)')
# plt.ylabel(r'Height above river (m)')
# plt.legend(fontsize=7)
# plt.title('a) MR', loc='left')

# plt.subplot(232)
# plt.semilogy(dt, np.cumsum(np.mean(MR_BR_bin1,axis=1))+np.cumsum(np.mean(MR_BR_bin2,axis=1))+np.cumsum(np.mean(MR_BR_bin3,axis=1))+np.cumsum(np.mean(MR_BR_bin4,axis=1)),  label='h <3.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin5,axis=1)), label='3.5m<= h <4.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin6,axis=1)), label='4.5m<= h <5.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin7,axis=1)), label='5.5m<= h <6.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin8,axis=1)), label='6.5m<= h <7.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin8,axis=1)), label='7.5m<= h <8.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin10,axis=1)), label='9.5m<= h <9.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin11,axis=1)), label='9.5m<= h <10.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin12,axis=1)), label='10.5m<= h <11.5m')
# plt.plot(dt, np.cumsum(np.mean(MR_BR_bin13,axis=1))+np.cumsum(np.mean(MR_BR_bin14,axis=1)), label='h>11.5m')
# # plt.plot(dt, np.cumsum(np.mean(MR_BR_bin4,axis=1)+np.mean(MR_BR_bin5,axis=1)+np.mean(MR_BR_bin6,axis=1)+np.mean(MR_BR_bin7,axis=1)+np.mean(MR_BR_bin8,axis=1)), 'k', lw=2, label='all bins')
# plt.legend(fontsize=7)
# plt.ylabel(r'Cumulative sum of sediment (m$^2$)')
# plt.title('b) MR', loc='left')

# plt.subplot(233)
# # plt.semilogy(MR, np.cumsum(np.mean(MR_BR_bin1,axis=0))+np.cumsum(np.mean(MR_BR_bin2,axis=0))+np.cumsum(np.mean(MR_BR_bin3,axis=0))+np.cumsum(np.mean(MR_BR_bin4,axis=0)), label='h <3.5m')
# # plt.plot(MR, np.cumsum(np.mean(MR_BR_bin5,axis=0)),  label='3.5m<= h <4.5m')
# # plt.plot(MR, np.cumsum(np.mean(MR_BR_bin6,axis=0)),label='4.5m<= h <5.5m')
# # plt.plot(MR, np.cumsum(np.mean(MR_BR_bin7,axis=0)), label='5.5m<= h <6.5m')
# # plt.plot(MR, np.cumsum(np.mean(MR_BR_bin8,axis=0)), label='6.5m<= h <8m')
# # plt.plot(MR, np.cumsum(np.mean(MR_BR_bin4,axis=0)+np.mean(MR_BR_bin5,axis=0)+np.mean(MR_BR_bin6,axis=0)+np.mean(MR_BR_bin7,axis=0)+np.mean(MR_BR_bin8,axis=0)), 'k', lw=2, label='all bins')
# plt.semilogy(MR, np.cumsum(np.mean(MR_BR_bin1,axis=0))+np.cumsum(np.mean(MR_BR_bin2,axis=0))+np.cumsum(np.mean(MR_BR_bin3,axis=0))+np.cumsum(np.mean(MR_BR_bin4,axis=0)),  label='h <3.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin5,axis=0)), label='3.5m<= h <4.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin6,axis=0)), label='4.5m<= h <5.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin7,axis=0)), label='5.5m<= h <6.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin8,axis=0)), label='6.5m<= h <7.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin8,axis=0)), label='7.5m<= h <8.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin10,axis=0)), label='9.5m<= h <9.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin11,axis=0)), label='9.5m<= h <10.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin12,axis=0)), label='10.5m<= h <11.5m')
# plt.plot(MR, np.cumsum(np.mean(MR_BR_bin13,axis=0))+np.cumsum(np.mean(MR_BR_bin14,axis=0)), label='h>11.5m')

# plt.legend(fontsize=7)
# plt.ylabel(r'Cumulative sum of sediment (m$^2$)')
# plt.xlabel("Distance downstream (km)")
# plt.title('c) MR', loc='left')


# plt.subplot(234)
# y = [np.mean(np.vstack(LR_BR_bin1)/grid2sqm),np.mean(np.vstack(LR_BR_bin2)/grid2sqm),np.mean(np.vstack(LR_BR_bin3)/grid2sqm),np.mean(np.vstack(LR_BR_bin4)/grid2sqm), np.mean(np.vstack(LR_BR_bin5)/grid2sqm), np.mean(np.vstack(LR_BR_bin6)/grid2sqm), np.mean(np.vstack(LR_BR_bin7)/grid2sqm), np.mean(np.vstack(LR_BR_bin8)/grid2sqm),np.mean(np.vstack(LR_BR_bin9)/grid2sqm),np.mean(np.vstack(LR_BR_bin10)/grid2sqm),np.mean(np.vstack(LR_BR_bin11)/grid2sqm),np.mean(np.vstack(LR_BR_bin12)/grid2sqm),np.mean(np.vstack(LR_BR_bin13)/grid2sqm),np.mean(np.vstack(LR_BR_bin14)/grid2sqm)]

# y2 = [np.std(np.vstack(LR_BR_bin1)/grid2sqm),np.std(np.vstack(LR_BR_bin2)/grid2sqm),np.std(np.vstack(LR_BR_bin3)/grid2sqm),np.std(np.vstack(LR_BR_bin4)/grid2sqm), np.std(np.vstack(LR_BR_bin5)/grid2sqm), np.std(np.vstack(LR_BR_bin6)/grid2sqm), np.std(np.vstack(LR_BR_bin7)/grid2sqm), np.std(np.vstack(LR_BR_bin8)/grid2sqm),np.std(np.vstack(LR_BR_bin9)/grid2sqm),np.std(np.vstack(LR_BR_bin10)/grid2sqm),np.std(np.vstack(LR_BR_bin11)/grid2sqm),np.std(np.vstack(LR_BR_bin12)/grid2sqm),np.std(np.vstack(LR_BR_bin13)/grid2sqm),np.std(np.vstack(LR_BR_bin14)/grid2sqm)]

# plt.semilogx(y,x,'k-o', label='Mean')
# plt.plot(y2,x,'r-s',label='Stdev.')
# # plt.plot(np.array(y2)/np.array(y),x,'b-x',label='CoV')
# plt.xlabel(r'Sediment area (m$^2$)')
# plt.ylabel(r'Height above river (m)')
# plt.legend(fontsize=7)
# plt.title('d) LR', loc='left')


# plt.subplot(235)
# plt.semilogy(dt, np.cumsum(np.mean(LR_BR_bin1,axis=1))+np.cumsum(np.mean(LR_BR_bin2,axis=1))+np.cumsum(np.mean(LR_BR_bin3,axis=1))+np.cumsum(np.mean(LR_BR_bin4,axis=1)),  label='h <3.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin5,axis=1)), label='3.5m<= h <4.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin6,axis=1)), label='4.5m<= h <5.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin7,axis=1)), label='5.5m<= h <6.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin8,axis=1)), label='6.5m<= h <7.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin8,axis=1)), label='7.5m<= h <8.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin10,axis=1)), label='9.5m<= h <9.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin11,axis=1)), label='9.5m<= h <10.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin12,axis=1)), label='10.5m<= h <11.5m')
# plt.plot(dt, np.cumsum(np.mean(LR_BR_bin13,axis=1))+np.cumsum(np.mean(LR_BR_bin14,axis=1)), label='h>11.5m')
# plt.title('e) LR', loc='left')
# plt.ylabel(r'Cumulative sum of sediment (m$^2$)')

# plt.subplot(236)
# plt.semilogy(LR, np.cumsum(np.mean(LR_BR_bin1,axis=0))+np.cumsum(np.mean(LR_BR_bin2,axis=0))+np.cumsum(np.mean(LR_BR_bin3,axis=0))+np.cumsum(np.mean(LR_BR_bin4,axis=0)),  label='h <3.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin5,axis=0)), label='3.5m<= h <4.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin6,axis=0)), label='4.5m<= h <5.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin7,axis=0)), label='5.5m<= h <6.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin8,axis=0)), label='6.5m<= h <7.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin8,axis=0)), label='7.5m<= h <8.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin10,axis=0)), label='9.5m<= h <9.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin11,axis=0)), label='9.5m<= h <10.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin12,axis=0)), label='10.5m<= h <11.5m')
# plt.plot(LR, np.cumsum(np.mean(LR_BR_bin13,axis=0))+np.cumsum(np.mean(LR_BR_bin14,axis=0)), label='h>11.5m')

# # plt.plot(LR, np.cumsum(np.mean(LR_BR_bin4,axis=0)+np.mean(LR_BR_bin5,axis=0)+np.mean(LR_BR_bin6,axis=0)+np.mean(LR_BR_bin7,axis=0)+np.mean(LR_BR_bin8,axis=0)), 'k', lw=2, label='all bins')
# plt.legend(fontsize=7)
# plt.ylabel(r'Cumulative sum of sediment (m$^2$)')
# plt.xlabel("Distance downstream (km)")
# plt.title('f) LR', loc='left')

# # plt.show()
# # plt.savefig("LR_wood_spacetime_plots_binned_height.png", dpi=300, bbox_inches="tight")
# plt.savefig("summaries/MR_LR_sediment_spacetime_plots_binned_height.png", dpi=300, bbox_inches="tight")

# plt.close()

####################################################

