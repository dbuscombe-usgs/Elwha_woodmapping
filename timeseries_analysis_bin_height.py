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

n_workers = 20
threads_per_worker = 2
memory_limit='50GB'

cwd = os.getcwd()
# run_bash = True

## factor that converts grid uints 1/8 x 1/8
# into units 1 x 1, i.e. 8 x 8
grid2sqm = 64

## we estimate over-ditizization factor
overdig_factor = 1.2


#############################################################
## start client
client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

# Create variable used for time axis
time_var = xr.Variable('time',times)


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
wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/Elwha_LR_*wood_filtered_bin0.1_regrid_final.tif'))
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
wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/Elwha_MR_*wood_filtered_bin0.1_regrid_final.tif'))
print(len(wood_files))

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')

# Rename the variable to a more useful name
MRwood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

#############################################################
# get mean detrended dems

####### MR
MR_detrend_dem = rioxarray.open_rasterio("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_global_regrid.tif", chunks=chunksize, dtype=dtype)
MR_detrend_dem = MR_detrend_dem.to_dataset('band').persist()
print(MR_detrend_dem.dims)

## remove height offset
MR_detrend_dem[1] = MR_detrend_dem[1] - MR_detrend_dem[1].min()

####### LR
LR_detrend_dem = rioxarray.open_rasterio("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/LR_DEM_detrend_global_regrid.tif", chunks=chunksize, dtype=dtype)
LR_detrend_dem = LR_detrend_dem.to_dataset('band').persist()
print(LR_detrend_dem.dims)

## remove height offset
LR_detrend_dem[1] = LR_detrend_dem[1] - LR_detrend_dem[1].min()

print(MRwood_geotiffs_ds.dims)

print(LRwood_geotiffs_ds.dims)

dists = pd.read_csv('br_dists.csv')
LR = np.hstack((0,np.array(dists['LR'])))
MR = np.hstack((0,np.array(dists['MR'][:43])))


#######################################################

# sum wood pixels in each BR reach, timestamp, and 4 height bins

MR_BR_bin1=[]
MR_BR_bin2=[]
MR_BR_bin3=[]
MR_BR_bin4=[]
MR_BR_bin5=[]
MR_BR_bin6=[]
MR_BR_bin7=[]
MR_BR_bin8=[]
for time in times:
    tmp = MRwood_geotiffs_ds.wood.sel(time=time)

    bin1=[]
    bin2=[]
    bin3=[]
    bin4=[]
    bin5=[]
    bin6=[]
    bin7=[]
    bin8=[]
    for g in tqdm(MRbudget_reaches_redo):
        wood_c = tmp.rio.clip([g], tmp.rio.crs)
        dem_tmp = MR_detrend_dem[1].rio.clip([g], MR_detrend_dem[1].rio.crs)

        result1 = (wood_c.where((dem_tmp < 4))).sum().compute().to_numpy() 
        bin1.append(float(result1))

        result2 = (wood_c.where((dem_tmp >= 4) & (dem_tmp < 5))).sum().compute().to_numpy() 
        bin2.append(float(result2))

        result3 = (wood_c.where((dem_tmp >= 5) & (dem_tmp < 6))).sum().compute().to_numpy() 
        bin3.append(float(result3))

        result4 = (wood_c.where((dem_tmp >= 6) & (dem_tmp < 7))).sum().compute().to_numpy() 
        bin4.append(float(result4))

        result5 = (wood_c.where((dem_tmp >= 7) & (dem_tmp < 8))).sum().compute().to_numpy() 
        bin5.append(float(result5))

        result6 = (wood_c.where((dem_tmp >= 8) & (dem_tmp < 9))).sum().compute().to_numpy() 
        bin6.append(float(result6))

        result7 = (wood_c.where((dem_tmp >= 9) & (dem_tmp < 10))).sum().compute().to_numpy() 
        bin7.append(float(result7))

        result8 = (wood_c.where((dem_tmp > 10))).sum().compute().to_numpy() 
        bin8.append(float(result8))

    MR_BR_bin1.append(bin1)
    MR_BR_bin2.append(bin2)
    MR_BR_bin3.append(bin3)
    MR_BR_bin4.append(bin4)
    MR_BR_bin5.append(bin5)
    MR_BR_bin6.append(bin6)
    MR_BR_bin7.append(bin7)
    MR_BR_bin8.append(bin8)

    print(MR_BR_bin5)


#######################################################

# LR_BR=[]
# for time in times:
#     tmp = LRwood_geotiffs_ds.wood.sel(time=time)
#     for g in tqdm(LRbudget_reaches_redo):
#         wood_c = tmp.rio.clip([g], tmp.rio.crs)
#         result = (wood_c).sum().compute().to_numpy() 
#         LR_BR.append(float(result))


# sum wood pixels in each BR reach, timestamp, and 4 height bins

LR_BR_bin1=[]
LR_BR_bin2=[]
LR_BR_bin3=[]
LR_BR_bin4=[]
LR_BR_bin5=[]
LR_BR_bin6=[]
LR_BR_bin7=[]
LR_BR_bin8=[]
for time in times:
    tmp = LRwood_geotiffs_ds.wood.sel(time=time)

    bin1=[]
    bin2=[]
    bin3=[]
    bin4=[]
    bin5=[]
    bin6=[]
    bin7=[]
    bin8=[]
    for g in tqdm(LRbudget_reaches_redo):
        wood_c = tmp.rio.clip([g], tmp.rio.crs)
        dem_tmp = LR_detrend_dem[1].rio.clip([g], LR_detrend_dem[1].rio.crs)

        result1 = (wood_c.where((dem_tmp < 4))).sum().compute().to_numpy() 
        bin1.append(float(result1))

        result2 = (wood_c.where((dem_tmp >= 4) & (dem_tmp < 5))).sum().compute().to_numpy() 
        bin2.append(float(result2))

        result3 = (wood_c.where((dem_tmp >= 5) & (dem_tmp < 6))).sum().compute().to_numpy() 
        bin3.append(float(result3))

        result4 = (wood_c.where((dem_tmp >= 6) & (dem_tmp < 7))).sum().compute().to_numpy() 
        bin4.append(float(result4))

        result5 = (wood_c.where((dem_tmp >= 7) & (dem_tmp < 8))).sum().compute().to_numpy() 
        bin5.append(float(result5))

        result6 = (wood_c.where((dem_tmp >= 8) & (dem_tmp < 9))).sum().compute().to_numpy() 
        bin6.append(float(result6))

        result7 = (wood_c.where((dem_tmp >= 9) & (dem_tmp < 10))).sum().compute().to_numpy() 
        bin7.append(float(result7))

        result8 = (wood_c.where((dem_tmp > 10))).sum().compute().to_numpy() 
        bin8.append(float(result8))

    LR_BR_bin1.append(bin1)
    LR_BR_bin2.append(bin2)
    LR_BR_bin3.append(bin3)
    LR_BR_bin4.append(bin4)
    LR_BR_bin5.append(bin5)
    LR_BR_bin6.append(bin6)
    LR_BR_bin7.append(bin7)
    LR_BR_bin8.append(bin8)

    print(LR_BR_bin5)


bins = [2,4.5,5.5,6.5,7.5,8.5,9.5,11]

MR_BR_bin1_scaled = np.vstack(MR_BR_bin1)/grid2sqm/A_MR
MR_BR_bin2_scaled = np.vstack(MR_BR_bin2)/grid2sqm/A_MR
MR_BR_bin3_scaled = np.vstack(MR_BR_bin3)/grid2sqm/A_MR
MR_BR_bin4_scaled = np.vstack(MR_BR_bin4)/grid2sqm/A_MR
MR_BR_bin5_scaled = np.vstack(MR_BR_bin5)/grid2sqm/A_MR
MR_BR_bin6_scaled = np.vstack(MR_BR_bin6)/grid2sqm/A_MR
MR_BR_bin7_scaled = np.vstack(MR_BR_bin7)/grid2sqm/A_MR
MR_BR_bin8_scaled = np.vstack(MR_BR_bin8)/grid2sqm/A_MR


A_MR = np.array(A_MR)
A_LR = np.array(A_LR)

########################################
plt.figure(figsize=(16,8))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(131)
y = [np.mean(np.vstack(MR_BR_bin4)/grid2sqm), np.mean(np.vstack(MR_BR_bin5)/grid2sqm), np.mean(np.vstack(MR_BR_bin6)/grid2sqm), np.mean(np.vstack(MR_BR_bin7)/grid2sqm), np.mean(np.vstack(MR_BR_bin8)/grid2sqm)]
y2 = [np.std(np.vstack(MR_BR_bin4)/grid2sqm), np.std(np.vstack(MR_BR_bin5)/grid2sqm), np.std(np.vstack(MR_BR_bin6)/grid2sqm), np.std(np.vstack(MR_BR_bin7)/grid2sqm), np.std(np.vstack(MR_BR_bin8)/grid2sqm)]

x=np.array([6.5,7.5,8.5,9.5,11])-3
plt.semilogx(y,x,'k-o', label='Mean')
plt.plot(y2,x,'r-s',label='Stdev.')
plt.plot(np.array(y2)/np.array(y),x,'b-x',label='CoV')
plt.xlabel(r'Wood area (m$^2$)')
plt.ylabel(r'Height above river at time of survey (m)')
plt.legend()
plt.title('a)', loc='left')

plt.subplot(132)
plt.semilogy(dt, np.mean(MR_BR_bin4_scaled,axis=1),  label='3.5m')
plt.plot(dt, np.mean(MR_BR_bin5_scaled,axis=1), label='4.5m')
plt.plot(dt, np.mean(MR_BR_bin6_scaled,axis=1), label='5.5m')
plt.plot(dt, np.mean(MR_BR_bin7_scaled,axis=1), label='6.5m')
plt.plot(dt, np.mean(MR_BR_bin8_scaled,axis=1), label='8m')
plt.plot(dt, np.mean(MR_BR_bin4_scaled,axis=1)+np.mean(MR_BR_bin5_scaled,axis=1)+np.mean(MR_BR_bin6_scaled,axis=1)+np.mean(MR_BR_bin7_scaled,axis=1)+np.mean(MR_BR_bin8_scaled,axis=1), 'k', lw=2, label='all bins')
plt.legend()
plt.ylabel(r'Mean wood concentration (m$^2$/m$^2$)')
plt.title('b)', loc='left')

plt.subplot(133)
plt.plot(MR, np.cumsum(np.mean(MR_BR_bin4_scaled,axis=0)),  label='3.5m')
plt.plot(MR, np.cumsum(np.mean(MR_BR_bin5_scaled,axis=0)), label='4.5m')
plt.plot(MR, np.cumsum(np.mean(MR_BR_bin6_scaled,axis=0)), label='5.5m')
plt.plot(MR, np.cumsum(np.mean(MR_BR_bin7_scaled,axis=0)), label='6.5m')
plt.plot(MR, np.cumsum(np.mean(MR_BR_bin8_scaled,axis=0)), label='8m')
plt.plot(MR, np.cumsum(np.mean(MR_BR_bin4_scaled,axis=0)+np.mean(MR_BR_bin5_scaled,axis=0)+np.mean(MR_BR_bin6_scaled,axis=0)+np.mean(MR_BR_bin7_scaled,axis=0)+np.mean(MR_BR_bin8_scaled,axis=0)), 'k', lw=2, label='all bins')
plt.legend()
plt.ylabel(r'Cumulative sum of mean wood concentration (m$^2$/m$^2$)')
plt.xlabel("Distance downstream (km)")
plt.title('c)', loc='left')

# plt.show()
plt.savefig("wood_spacetime_plots_binned_height.png", dpi=300, bbox_inches="tight")
plt.close()





LR_BR_bin1_scaled = np.vstack(LR_BR_bin1)/grid2sqm/A_LR
LR_BR_bin2_scaled = np.vstack(LR_BR_bin2)/grid2sqm/A_LR
LR_BR_bin3_scaled = np.vstack(LR_BR_bin3)/grid2sqm/A_LR
LR_BR_bin4_scaled = np.vstack(LR_BR_bin4)/grid2sqm/A_LR
LR_BR_bin5_scaled = np.vstack(LR_BR_bin5)/grid2sqm/A_LR
LR_BR_bin6_scaled = np.vstack(LR_BR_bin6)/grid2sqm/A_LR
LR_BR_bin7_scaled = np.vstack(LR_BR_bin7)/grid2sqm/A_LR
LR_BR_bin8_scaled = np.vstack(LR_BR_bin8)/grid2sqm/A_LR



########################################
plt.figure(figsize=(16,8))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(131)
y = [np.mean(np.vstack(LR_BR_bin4)/grid2sqm), np.mean(np.vstack(LR_BR_bin5)/grid2sqm), np.mean(np.vstack(LR_BR_bin6)/grid2sqm), np.mean(np.vstack(LR_BR_bin7)/grid2sqm), np.mean(np.vstack(LR_BR_bin8)/grid2sqm)]
y2 = [np.std(np.vstack(LR_BR_bin4)/grid2sqm), np.std(np.vstack(LR_BR_bin5)/grid2sqm), np.std(np.vstack(LR_BR_bin6)/grid2sqm), np.std(np.vstack(LR_BR_bin7)/grid2sqm), np.std(np.vstack(LR_BR_bin8)/grid2sqm)]

x=np.array([6.5,7.5,8.5,9.5,11])-3
plt.semilogx(y,x,'k-o', label='Mean')
plt.plot(y2,x,'r-s',label='Stdev.')
plt.plot(np.array(y2)/np.array(y),x,'b-x',label='CoV')
plt.xlabel(r'Wood area (m$^2$)')
plt.ylabel(r'Height above river at time of survey (m)')
plt.legend()
plt.title('a)', loc='left')

plt.subplot(132)
plt.semilogy(dt, np.mean(LR_BR_bin4_scaled,axis=1),  label='3.5m')
plt.plot(dt, np.mean(LR_BR_bin5_scaled,axis=1), label='4.5m')
plt.plot(dt, np.mean(LR_BR_bin6_scaled,axis=1), label='5.5m')
plt.plot(dt, np.mean(LR_BR_bin7_scaled,axis=1), label='6.5m')
plt.plot(dt, np.mean(LR_BR_bin8_scaled,axis=1), label='8m')
plt.plot(dt, np.mean(LR_BR_bin4_scaled,axis=1)+np.mean(LR_BR_bin5_scaled,axis=1)+np.mean(LR_BR_bin6_scaled,axis=1)+np.mean(LR_BR_bin7_scaled,axis=1)+np.mean(LR_BR_bin8_scaled,axis=1), 'k', lw=2, label='all bins')
plt.legend()
plt.ylabel(r'Mean wood concentration (m$^2$/m$^2$)')
plt.title('b)', loc='left')

plt.subplot(133)
plt.plot(LR, np.cumsum(np.mean(LR_BR_bin4_scaled,axis=0)),  label='3.5m')
plt.plot(LR, np.cumsum(np.mean(LR_BR_bin5_scaled,axis=0)), label='4.5m')
plt.plot(LR, np.cumsum(np.mean(LR_BR_bin6_scaled,axis=0)), label='5.5m')
plt.plot(LR, np.cumsum(np.mean(LR_BR_bin7_scaled,axis=0)), label='6.5m')
plt.plot(LR, np.cumsum(np.mean(LR_BR_bin8_scaled,axis=0)), label='8m')
plt.plot(LR, np.cumsum(np.mean(LR_BR_bin4_scaled,axis=0)+np.mean(LR_BR_bin5_scaled,axis=0)+np.mean(LR_BR_bin6_scaled,axis=0)+np.mean(LR_BR_bin7_scaled,axis=0)+np.mean(LR_BR_bin8_scaled,axis=0)), 'k', lw=2, label='all bins')
plt.legend()
plt.ylabel(r'Cumulative sum of mean wood concentration (m$^2$/m$^2$)')
plt.xlabel("Distance downstream (km)")
plt.title('c)', loc='left')

# plt.show()
plt.savefig("LR_wood_spacetime_plots_binned_height.png", dpi=300, bbox_inches="tight")
plt.close()



####################################################


np.savez('Wood_time_series_bins_height_MR.npz', MR_BR_bin4_scaled = MR_BR_bin4_scaled, MR_BR_bin5_scaled = MR_BR_bin5_scaled, MR_BR_bin6_scaled=MR_BR_bin6_scaled, MR_BR_bin7_scaled = MR_BR_bin7_scaled, MR_BR_bin8_scaled=MR_BR_bin8_scaled, MR_BR_bin4=MR_BR_bin4, MR_BR_bin5=MR_BR_bin5, MR_BR_bin6=MR_BR_bin6, MR_BR_bin7=MR_BR_bin7, MR_BR_bin8=MR_BR_bin8)


np.savez('Wood_time_series_bins_height_LR.npz', LR_BR_bin4_scaled = LR_BR_bin4_scaled, LR_BR_bin5_scaled = LR_BR_bin5_scaled, LR_BR_bin6_scaled=LR_BR_bin6_scaled, LR_BR_bin7_scaled = LR_BR_bin7_scaled, LR_BR_bin8_scaled=LR_BR_bin8_scaled, LR_BR_bin4=LR_BR_bin4, LR_BR_bin5=LR_BR_bin5, LR_BR_bin6=LR_BR_bin6, LR_BR_bin7=LR_BR_bin7, LR_BR_bin8=LR_BR_bin8)



# plt.subplot(221)
# plt.plot(MR_BR_bin4_scaled, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('a) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(222)
# plt.imshow(MR_BR_bin5_scaled, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(223)
# plt.imshow(MR_BR_bin6_scaled, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('c) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(224)
# plt.imshow(MR_BR_bin7_scaled, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('d) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(224)
# plt.imshow(MR_BR_bin8_scaled, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('d) ', loc='left'); plt.xlabel("Distance downstream (km)"); 


# LR_BRarr = np.vstack(LR_BR).reshape(len(times),-1)/grid2sqm
# MR_BRarr = np.vstack(MR_BR).reshape(len(times),-1)/grid2sqm

