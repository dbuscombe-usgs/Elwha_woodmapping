
## Dan Buscombe, Marda Science
## 2023

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
from skimage.measure import label, regionprops_table
from pointpats import centrography, distance_statistics
import geopandas as gpd

try:
    from skimage.measure import pearson_corr_coeff, intersection_coeff, manders_coloc_coeff, manders_overlap_coeff, pearson_corr_coeff
except:
    print("you need scikit-image>=0.21.0")
    sys.exit(2)

from scipy.signal import convolve2d


# ##========================================================
def inpaint_nans(im):
    ipn_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # kernel for inpaint_nans
    nans = np.isnan(im)
    while np.sum(nans) > 0:
        im[nans] = 0
        vNeighbors = convolve2d(
            (nans == False), ipn_kernel, mode="same", boundary="symm"
        )
        im2 = convolve2d(im, ipn_kernel, mode="same", boundary="symm")
        im2[vNeighbors > 0] = im2[vNeighbors > 0] / vNeighbors[vNeighbors > 0]
        im2[vNeighbors == 0] = np.nan
        im2[(nans == False)] = im[(nans == False)]
        im = im2
        nans = np.isnan(im)
    return im


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

#############################################################
# # start client
n_workers = 20
threads_per_worker = 2
memory_limit='20GB'

client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

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


dists = pd.read_csv('br_dists.csv')
LR = np.hstack((0,np.array(dists['LR'])))
MR = np.hstack((0,np.array(dists['MR'][:43])))

LR = rescale_array(LR,11,2)
MR = rescale_array(MR[::-1],12,20)

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

dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_2*.tif'))
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
MR_dem_detrend_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})


dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/LR_DEM_detrend_2*.tif'))
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
LR_dem_detrend_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})


#############################################


#### MR
MR_MO=[]; MR_MC = []; MR_IC = []; MR_PC = []; MR_PV=[]
for counter,time in enumerate(times):
    print(counter)
    if counter>0:
        tmp = MRwood_geotiffs_ds.wood.sel(time=times[counter])
        tmp0 = MRwood_geotiffs_ds.wood.sel(time=times[counter-1])
       
        for g in tqdm(MRbudget_reaches_redo):
            wood_c = tmp.rio.clip([g], tmp.rio.crs)
            wood_c0 = tmp0.rio.clip([g], tmp.rio.crs)

            MR_IC.append(intersection_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
            MR_MO.append(manders_overlap_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
            MR_MC.append(manders_coloc_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
            pcc, pval = pearson_corr_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1)
            MR_PC.append(pcc)
            MR_PV.append(pval)

MR_IC = np.array(MR_IC).reshape(len(times)-1,len(MRbudget_reaches_redo))
MR_MO = np.array(MR_MO).reshape(len(times)-1,len(MRbudget_reaches_redo))
MR_MC = np.array(MR_MC).reshape(len(times)-1,len(MRbudget_reaches_redo))
MR_PC = np.array(MR_PC).reshape(len(times)-1,len(MRbudget_reaches_redo))
MR_PV = np.array(MR_PV).reshape(len(times)-1,len(MRbudget_reaches_redo))

np.savez('summaries/MR_association_metrics_surveypairs.npz', MR_IC = MR_IC, MR_MO = MR_MO, MR_MC = MR_MC, MR_PC = MR_PC, MR_PV=MR_PV)


#### LR
LR_MO=[]; LR_MC = []; LR_IC = []; LR_PC = []; LR_PV=[]
for counter,time in enumerate(times):
    print(counter)
    if counter>0:
        tmp = LRwood_geotiffs_ds.wood.sel(time=times[counter])
        tmp0 = LRwood_geotiffs_ds.wood.sel(time=times[counter-1])
       
        for g in tqdm(LRbudget_reaches_redo):
            wood_c = tmp.rio.clip([g], tmp.rio.crs)
            wood_c0 = tmp0.rio.clip([g], tmp.rio.crs)

            LR_IC.append(intersection_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
            LR_MO.append(manders_overlap_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
            LR_MC.append(manders_coloc_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
            pcc, pval = pearson_corr_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1)
            LR_PC.append(pcc)
            LR_PV.append(pval)

LR_IC = np.array(LR_IC).reshape(len(times)-1,len(LRbudget_reaches_redo))
LR_MO = np.array(LR_MO).reshape(len(times)-1,len(LRbudget_reaches_redo))
LR_MC = np.array(LR_MC).reshape(len(times)-1,len(LRbudget_reaches_redo))
LR_PC = np.array(LR_PC).reshape(len(times)-1,len(LRbudget_reaches_redo))
LR_PV = np.array(LR_PV).reshape(len(times)-1,len(LRbudget_reaches_redo))

np.savez('summaries/LR_association_metrics_surveypairs.npz', LR_IC = LR_IC, LR_MO = LR_MO, LR_MC = LR_MC, LR_PC = LR_PC, LR_PV=LR_PV)



#### MR
wood0 = MRwood_geotiffs_ds.wood.sel(time=times[0])

MR_MO=[]; MR_MC = []; MR_IC = []; MR_PC = []; MR_PV=[]
for time in times[1:]:
    tmp = MRwood_geotiffs_ds.wood.sel(time=time)
    for g in tqdm(MRbudget_reaches_redo):
        wood_c = tmp.rio.clip([g], tmp.rio.crs)
        wood_c0 = wood0.rio.clip([g], tmp.rio.crs)

        MR_IC.append(intersection_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
        MR_MO.append(manders_overlap_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
        MR_MC.append(manders_coloc_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
        pcc, pval = pearson_corr_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1)
        MR_PC.append(pcc)
        MR_PV.append(pval)

#### LR
wood0 = LRwood_geotiffs_ds.wood.sel(time=times[0])

LR_MO=[]; LR_MC = []; LR_IC = []; LR_PC=[]; LR_PV=[]
for time in times[1:]:
    tmp = LRwood_geotiffs_ds.wood.sel(time=time)
    for g in tqdm(LRbudget_reaches_redo):
        wood_c = tmp.rio.clip([g], tmp.rio.crs)
        wood_c0 = wood0.rio.clip([g], tmp.rio.crs)

        LR_IC.append(intersection_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
        LR_MO.append(manders_overlap_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
        LR_MC.append(manders_coloc_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1))
        pcc, pval = pearson_corr_coeff(wood_c0.to_numpy()==1, wood_c.to_numpy()==1)
        LR_PC.append(pcc)
        LR_PV.append(pval)


LR_IC = np.array(LR_IC).reshape(len(times)-1,len(LRbudget_reaches_redo))
LR_MO = np.array(LR_MO).reshape(len(times)-1,len(LRbudget_reaches_redo))
LR_MC = np.array(LR_MC).reshape(len(times)-1,len(LRbudget_reaches_redo))
LR_PC = np.array(LR_PC).reshape(len(times)-1,len(LRbudget_reaches_redo))
LR_PV = np.array(LR_PV).reshape(len(times)-1,len(LRbudget_reaches_redo))

MR_IC = np.array(MR_IC).reshape(len(times)-1,len(MRbudget_reaches_redo))
MR_MO = np.array(MR_MO).reshape(len(times)-1,len(MRbudget_reaches_redo))
MR_MC = np.array(MR_MC).reshape(len(times)-1,len(MRbudget_reaches_redo))
MR_PC = np.array(MR_PC).reshape(len(times)-1,len(MRbudget_reaches_redo))
MR_PV = np.array(MR_PV).reshape(len(times)-1,len(MRbudget_reaches_redo))


np.savez('summaries/LR_association_metrics.npz', LR_IC = LR_IC, LR_MO = LR_MO, LR_MC = LR_MC, LR_PC = LR_PC, LR_PV=LR_PV)
np.savez('summaries/MR_association_metrics.npz', MR_IC = MR_IC, MR_MO = MR_MO, MR_MC = MR_MC, MR_PC = MR_PC, MR_PV=LR_PV)



with np.load('summaries/LR_association_metrics.npz', allow_pickle=True) as f:
    LR_ICg = f['LR_IC']
    LR_MOg = f['LR_MO']
    LR_MCg = f['LR_MC']
    LR_PCg = f['LR_PC']
    LR_PVg = f['LR_PV']

with np.load('summaries/MR_association_metrics.npz', allow_pickle=True) as f:
    MR_ICg = f['MR_IC']
    MR_MOg = f['MR_MO']
    MR_MCg = f['MR_MC']
    MR_PCg = f['MR_PC']
    MR_PVg = f['MR_PV']



with np.load('summaries/LR_association_metrics_surveypairs.npz', allow_pickle=True) as f:
    LR_IC = f['LR_IC']
    LR_MO = f['LR_MO']
    LR_MC = f['LR_MC']
    LR_PC = f['LR_PC']
    LR_PV = f['LR_PV']

with np.load('summaries/MR_association_metrics_surveypairs.npz', allow_pickle=True) as f:
    MR_IC = f['MR_IC']
    MR_MO = f['MR_MO']
    MR_MC = f['MR_MC']
    MR_PC = f['MR_PC']
    MR_PV = f['MR_PV']



file = '../raw_data/GIS/LR_transects_clipped_active.geojson'
# file = '../raw_data/GIS/LR_active_widths.geojson'
LR_widths = gpd.read_file(file)
LR_width = LR_widths['width'].values
LR_full_width = LR_widths['full_width'].values

file = '../raw_data/GIS/MR_transects_clipped_active.geojson'
# file = '../raw_data/GIS/MR_active_widths.geojson'
MR_widths = gpd.read_file(file)
MR_width = MR_widths['width'].values
MR_full_width = MR_widths['full_width'].values


## plot width vbersus persistenmce


MR_PC2 = inpaint_nans(np.flipud(MR_PC))
LR_PC2 = inpaint_nans(np.flipud(LR_PC))

MR_PC2[MR_PC2<0] = 0
LR_PC2[LR_PC2<0] = 0



MR_PC2g = inpaint_nans(np.flipud(MR_PCg))
LR_PC2g = inpaint_nans(np.flipud(LR_PCg))

MR_PC2g[MR_PC2g<0] = 0
LR_PC2g[LR_PC2g<0] = 0

########################################
plt.figure(figsize=(16,12))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(231)
plt.imshow(MR_PC2g, cmap='inferno', extent=[MR[0], MR[-1] , dt[1], dt[-1]], aspect='auto', vmax=1, vmin=0)
# cb=plt.colorbar(); cb.set_label(r"Correlation coefficient")
plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
plt.title('a) ', loc='left'); plt.xlabel("River kilometer"); 

plt.subplot(232)
plt.imshow(LR_PC2g, cmap='inferno', extent=[LR[0], LR[-1] , dt[1], dt[-1]], aspect='auto', vmax=1, vmin=0)
# cb=plt.colorbar(); cb.set_label(r"Correlation coefficient")
plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.title('b) ', loc='left'); plt.xlabel("River kilometer"); 

plt.subplot(233)
plt.plot(np.nanmean(MR_PC2g,axis=1)[::-1],dt[1:],'k-', label='MR')
im=plt.plot(np.nanmean(LR_PC2g,axis=1)[::-1],dt[1:],'r--', label='LR')
plt.legend(fontsize=8)
plt.gca().invert_yaxis()
plt.title('b) ', loc='left'); plt.xlabel(r"Mean autocorrelation");
plt.xlim(0,.4)
# plt.colorbar(im)

plt.subplot(234)
im2=plt.plot(MR,np.nanmean(MR_PC2g,axis=0),'k',lw=2)
# cb=plt.colorbar();
# plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.title('c) ', loc='left'); plt.xlabel("River kilometer"); 
plt.ylim(0,.6)
plt.xlim(20,12)

plt.subplot(235)
im3=plt.plot(LR,np.nanmean(LR_PC2g,axis=0),'r--',lw=2)
# cb=plt.colorbar();
# plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
# plt.title('c) ', loc='left'); plt.xlabel("River kilometer"); 
plt.ylim(0,.6)
plt.xlim(11,2)

plt.subplot(236)
x = np.max(MR_PC2g,axis=0)
y = MR_width.copy()
x = x[x>0]
xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x)
plt.plot(y,xi,'ko', label='MR')
x = np.max(LR_PC2g,axis=0)
y = LR_width.copy()
x = x[x>0]
xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x)
# yi = np.interp(np.linspace(np.nanmin(y), np.nanmax(y),len(x)),np.linspace(0,len(y),len(y)),y)
plt.plot(y,xi,'rs', label='LR')
plt.title('d) ', loc='left'); 
plt.legend()
plt.ylabel("Maximum\nautocorrelation (-)"); plt.xlabel("Maximum active\nchannel width (m)");
# plt.ylim(0,.4)


plt.show()
# plt.savefig("summaries/wood_spacetime_persistence.png", dpi=300, bbox_inches="tight")
# plt.close()



sed_load = pd.read_csv('../raw_data/time_series/Elwha_DailySedimentLoads_2011to2016.csv')

sed_load = sed_load[['Day',
'Daily Discharge (m3/s)',
'Total sediment discharge (tonnes)',
'Ave fraction fines (based on two turbidimeters)']]


dt_sed = [datetime.strptime(time,'%m/%d/%Y') for time in sed_load['Day']]
dt_sed = np.array(dt_sed)

ind = np.argsort(dt_sed)

t_sed = np.array([float(d.strftime('%s')) for d in dt_sed[ind]])
t =  np.array([float(d.strftime('%s')) for d in dt])


# O_MR = np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values)
OS = np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values)

OQ = np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values)



########################################
plt.figure(figsize=(16,12))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(231)
plt.imshow(MR_PC2, cmap='inferno', extent=[MR[0], MR[-1] , dt[1], dt[-1]], aspect='auto', vmax=1, vmin=0)
# cb=plt.colorbar(); cb.set_label(r"Correlation coefficient")
plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
plt.title('a) ', loc='left'); plt.xlabel("River kilometer"); 

plt.subplot(232)
plt.imshow(LR_PC2, cmap='inferno', extent=[LR[0], LR[-1] , dt[1], dt[-1]], aspect='auto', vmax=1, vmin=0)
# cb=plt.colorbar(); cb.set_label(r"Correlation coefficient")
plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.title('b) ', loc='left'); plt.xlabel("River kilometer"); 

plt.subplot(233)
plt.plot(np.nanmean(MR_PC2,axis=1),dt[1:],'k-', label='MR')
im=plt.plot(np.nanmean(LR_PC2,axis=1),dt[1:],'r--', label='LR')
plt.legend(fontsize=8)
plt.gca().invert_yaxis()
plt.title('b) ', loc='left'); plt.xlabel(r"Median autocorrelation");
plt.xlim(0,.8)
# plt.colorbar(im)

plt.subplot(234)
im2=plt.plot(MR,np.nanmedian(MR_PC2,axis=0),'k',lw=2)
# cb=plt.colorbar();
# plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.title('c) ', loc='left'); plt.xlabel("River kilometer"); 
plt.ylim(0,.8)
plt.xlim(20,12)

plt.subplot(235)
im3=plt.plot(LR,np.nanmedian(LR_PC2,axis=0),'r--',lw=2)
# cb=plt.colorbar();
# plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
# plt.title('c) ', loc='left'); plt.xlabel("River kilometer"); 
plt.ylim(0,.8)
plt.xlim(11,2)

plt.subplot(236)
x = np.max(MR_PC2,axis=0)
y = MR_full_width.copy()
xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x, period=1)
plt.plot(y,xi,'ko', alpha=0.5, label='MR')
x = np.max(LR_PC2,axis=0)
y = LR_full_width.copy()
xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x, period=1)
plt.plot(y,xi,'rs', alpha=0.5, label='LR')

plt.title('d) ', loc='left'); 
plt.legend()
plt.ylabel("Maximum\nautocorrelation (-)"); plt.xlabel("Channel width (m)");
plt.ylim(0,1)

plt.savefig("summaries/wood_spacetime_persistence_pairwise_medians.png", dpi=300, bbox_inches="tight")
plt.close()



with np.load('../results/LR_spatial_metrics.npz', allow_pickle=True) as f:
    LR_GIO = f['LR_GIO']
    LR_GIR = f['LR_GIR']
    LR_CLUSTERED = f['LR_CLUSTERED']
    LR_GTEST = f['LR_GTEST']
    LR_FTEST=f['LR_FTEST']
    LR_FIO=f['LR_FIO']
    LR_FIR=f['LR_FIR']
    LR_DISPERSED=f['LR_DISPERSED']


F=[]
for k in LR_FTEST:
    try:
        F.append(k[1])
    except:
        F.append(np.ones(40)*np.nan)
    # try:
    #     F.append(k[1]) #[4][1])
    # except: 
    #     F.append(k[6][1])

LR_DISPERSED = np.array(LR_DISPERSED).reshape(len(times),len(LRbudget_reaches_redo))
LR_CLUSTERED = np.array(LR_CLUSTERED).reshape(len(times),len(LRbudget_reaches_redo))

# LR_FTEST = np.array(LR_FTEST).reshape(len(times),len(LRbudget_reaches_redo))
LR_FIO = np.array(LR_FIO).reshape(len(times),len(LRbudget_reaches_redo))
LR_FIR = np.array(LR_FIR).reshape(len(times),len(LRbudget_reaches_redo))

# LR_GTEST = np.array(LR_GTEST).reshape(len(times),len(LRbudget_reaches_redo))
LR_GIO = np.array(LR_GIO).reshape(len(times),len(LRbudget_reaches_redo))
LR_GIR = np.array(LR_GIR).reshape(len(times),len(LRbudget_reaches_redo))




with np.load('../results/MR_spatial_metrics.npz', allow_pickle=True) as f:
    MR_GIO = f['MR_GIO']
    MR_GIR = f['MR_GIR']
    MR_CLUSTERED = f['MR_CLUSTERED']
    MR_GTEST = f['MR_GTEST']
    MR_FTEST=f['MR_FTEST']
    MR_FIO=f['MR_FIO']
    MR_FIR=f['MR_FIR']
    MR_DISPERSED=f['MR_DISPERSED']


# FMR=[]
# for k in MR_FTEST:
#     try:
#         FMR.append(k[4][1])
#     except: 
#         FMR.append(k[6][1])

MR_DISPERSED = np.array(MR_DISPERSED).reshape(len(times),len(MRbudget_reaches_redo))
MR_CLUSTERED = np.array(MR_CLUSTERED).reshape(len(times),len(MRbudget_reaches_redo))

# MR_FTEST = np.array(MR_FTEST).reshape(len(times),len(MRbudget_reaches_redo))
MR_FIO = np.array(MR_FIO).reshape(len(times),len(MRbudget_reaches_redo))
MR_FIR = np.array(MR_FIR).reshape(len(times),len(MRbudget_reaches_redo))

# MR_GTEST = np.array(MR_GTEST).reshape(len(times),len(MRbudget_reaches_redo))
MR_GIO = np.array(MR_GIO).reshape(len(times),len(MRbudget_reaches_redo))
MR_GIR = np.array(MR_GIR).reshape(len(times),len(MRbudget_reaches_redo))



# with np.load('summaries/Wood_time_series.npz', allow_pickle=True) as f:
#     LR_BRarr = f['LR_BRarr']
#     MR_BRarr = f['MR_BRarr']
#     dt = f['dt']
#     grid2sqm = f['grid2sqm']

# dists = pd.read_csv('br_dists.csv')
# LR = np.hstack((0,np.array(dists['LR'])))
# MR = np.hstack((0,np.array(dists['MR'][:43])))

# #### divide out by area of each BR for a wood concentration\
# A_MR = np.array(A_MR)
# A_LR = np.array(A_LR)

# ## wood
# MR_BRarr_c = MR_BRarr/A_MR
# LR_BRarr_c = LR_BRarr/A_LR

dt = np.array(dt)



fig, ax = plt.subplots(nrows=2, ncols=3)
fig.set_size_inches(16,8)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

prop_disp = np.nansum(MR_DISPERSED,axis=1)/len(MRbudget_reaches_redo)
y = np.nanmean(MR_FIO,axis=-1)
y2 = np.nanmean(MR_FIR,axis=-1)

ax[0][0].plot(dt, y, 'k:', lw=3, label='Observed')
ax[0][0].plot(dt, y2, 'm--', label='Random')
ax[0][0].set_ylabel('Reach-averaged F-test statistic', color='m')
ax[0][0].set_title('a) MR', loc='left');
ax[0][0].legend(loc=3)
ax[0][0].set_ylim(0,.65)

ax20 = ax[0][0].twinx()
# ax2.plot(dt, prop_disp,'b')
ind = np.where(y>y2)[0]
ax20.plot(dt[ind], prop_disp[ind],'b-o')
ind = np.where(y<y2)[0]
ax20.plot(dt[ind], prop_disp[ind],'go')
ax20.set_ylabel('Proportion of reach dispersed', color='b')
ax20.set_ylim(0,.75)

prop_disp = np.nansum(LR_DISPERSED,axis=1)/len(LRbudget_reaches_redo)
y = np.nanmean(LR_FIO,axis=-1)
y2 = np.nanmean(LR_FIR,axis=-1)

ax[1][0].plot(dt, y, 'r:', lw=3, label='Observed')
ax[1][0].plot(dt, y2, 'm--', label='Random')
ax[1][0].set_ylabel('Reach-averaged F-test statistic', color='m')
ax[1][0].set_title('b) LR', loc='left');
ax[1][0].legend(loc=3)
ax[1][0].set_ylim(0,.65)

ax2 = ax[1][0].twinx()
# ax2.plot(dt, prop_disp,'b')
ind = np.where(y>y2)[0]
ax2.plot(dt[ind], prop_disp[ind],'b-o')
ind = np.where(y<y2)[0]
ax2.plot(dt[ind], prop_disp[ind],'go')
ax2.set_ylabel('Proportion of reach dispersed', color='b')
ax2.set_ylim(0,.75)

##### dfispersion versus persistence
ax[0][1].plot(np.nanmedian(MR_PC2,axis=0), np.nanmedian(MR_FIO,axis=0),'ko', label='MR')
ax[0][1].plot(np.nanmedian(LR_PC2,axis=0), np.nanmedian(LR_FIO,axis=0),'rs', label='LR')
ax[0][1].set_ylabel('Dispersion; median F-score (-)', color='k')
ax[0][1].set_xlabel('Persistence; median autocorrelation (-)', color='k')
ax[0][1].set_title('c)', loc='left');
ax[0][1].legend(loc=3)
ax[0][1].set_ylim(0,1.03)
ax[0][1].set_xlim(0,1.03)

##### dfispersion versus width
# x = np.nanmax(MR_FIO,axis=0)
x = np.nanmedian(MR_FIO,axis=0)
y = MR_full_width.copy()
xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x,period=1)
ax[1][1].plot(y, xi,'ko', label='MR')

# x = np.nanmax(LR_FIO,axis=0)
x = np.nanmedian(LR_FIO,axis=0)
y = LR_full_width.copy()
xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x,period=1)
ax[1][1].plot(y, xi,'rs', label='LR')

ax[1][1].set_ylabel('Dispersion; median F-score (-)', color='k')
ax[1][1].set_xlabel('Channel width (m)', color='k')
ax[1][1].set_title('d)', loc='left');
ax[1][1].legend(loc=4)
ax[1][1].set_ylim(0,1.03)
# ax[1][1].set_xlim(0,300)

##### dfispersion versus persistence
ax[0][2].plot(np.nanmax(MR_PC2,axis=0), np.nanmax(MR_FIO,axis=0),'ko', label='MR')
ax[0][2].plot(np.nanmax(LR_PC2,axis=0), np.nanmax(LR_FIO,axis=0),'rs', label='LR')
ax[0][2].set_ylabel('Dispersion; maximum F-score (-)', color='k')
ax[0][2].set_xlabel('Persistence; maximum autocorrelation (-)', color='k')
ax[0][2].set_title('e)', loc='left');
ax[0][2].legend(loc=3)
ax[0][2].set_ylim(0,1.03)
ax[0][2].set_xlim(0,1.03)

##### dfispersion versus width
x = np.nanmax(MR_FIO,axis=0)
# x = np.nanmedian(MR_FIO,axis=0)
y = MR_full_width.copy()
xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x,period=1)
ax[1][2].plot(y, xi,'ko', label='MR')

x = np.nanmax(LR_FIO,axis=0)
# x = np.nanmedian(LR_FIO,axis=0)
y = LR_full_width.copy()
xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x,period=1)
ax[1][2].plot(y, xi,'rs', label='LR')

ax[1][2].set_ylabel('Dispersion; maximum F-score (-)', color='k')
ax[1][2].set_xlabel('Channel width (m)', color='k')
ax[1][2].set_title('f)', loc='left');
ax[1][2].legend(loc=4)
ax[1][1].set_ylim(0,1.03)
# ax[1][1].set_xlim(0,300)

# plt.show()

plt.savefig("summaries/dispersion_stats.png", dpi=300, bbox_inches="tight")
plt.close()






######################################################
    
# MR_props=[]
# for time in times:
#     tmp = MRwood_geotiffs_ds.wood.sel(time=time)
#     for g in tqdm(MRbudget_reaches_redo):
#         wood_c = tmp.rio.clip([g], tmp.rio.crs)
#         label_img = label(wood_c==1)
#         props = regionprops_table(label_img, properties=('area','axis_minor_length','centroid','orientation', 'coords'))
#         MR_props.append(props)

#######################################################


#Centrography is the analysis of centrality in a point pattern. 
# By “centrality,” we mean the general location and dispersion of the pattern. 
# centrography is the point pattern equivalent of measures of central tendency such as the mean. 
# These measures are useful because they allow us to summarize spatial distributions in smaller sets of information (e.g., a single point). 
# Many different indices are used in centrography to provide an indication of “where” a point pattern is, 
# how tightly the point pattern clusters around its center, or how irregular its shape is.


### pip install pointpats
# Point Pattern Analysis
### https://geographicdata.science/book/notebooks/08_point_pattern_analysis.html
#Centrography is the analysis of centrality in a point pattern. 
# By “centrality,” we mean the general location and dispersion of the pattern. 
# centrography is the point pattern equivalent of measures of central tendency such as the mean. 
# These measures are useful because they allow us to summarize spatial distributions in smaller sets of information (e.g., a single point). 
# Many different indices are used in centrography to provide an indication of “where” a point pattern is, how tightly the point pattern clusters around its center, or how irregular its shape is.


## A measure of dispersion that is common in centrography is the standard distance. 
# This measure provides the average distance away from the center of the point cloud (such as measured by the center of mass).
# centrography.std_distance(coordinates)

# dispersion: whether points tend to all cluster near one another or disperse evenly throughout the area. 
# The first set of techniques, quadrat statistics, receive their name after their approach to split the data up into small areas (quadrants). Once created, these “buckets” are used to examine the uniformity of counts across them. The second set of techniques all derive from Ripley (1988) and involve measurements of the distance between points in a point pattern.

# Ripley’s alphabet of functions

# The second group of spatial statistics we consider focuses on the distributions of two quantities in a point pattern: nearest neighbor distances and what we will term “gaps” in the pattern. They derive from seminal work by [Rip91] on how to characterize clustering or co-location in point patterns. Each of these characterizes an aspect of the point pattern as we increase the distance range from each point to calculate them.

# The first function, Ripley’s
# , focuses on the distribution of nearest neighbor distances. That is, the
# function summarizes the distances between each point in the pattern and its nearest neighbor. 

# Ripley’s G keeps track of the proportion of points for which the nearest neighbor is within a given distance threshold, and plots that cumulative percentage against the increasing distance radii. The distribution of these cumulative percentages has a distinctive shape under completely spatially random processes. The intuition behind Ripley’s G goes as follows: we can learn about how similar our pattern is to a spatially random one by computing the cumulative distribution of nearest neighbor distances over increasing distance thresholds, and comparing it to that of a set of simulated patterns that follow a known spatially random process. Usually, a spatial Poisson point process is used as such reference distribution.



# # The second function we introduce is Ripley’s F. 
# # Where the G function works by analyzing the distance between points in the pattern, the F function works by analyzing the distance to points in the pattern from locations in empty space
# # it characterizes the typical distance from arbitrary points in empty space to the point pattern. 
# # More explicitly, the F accumulates, for a growing distance range, the percentage of points that can be found within that range from a random point pattern generated within the extent of the observed pattern. If the pattern has large gaps or empty areas, the F function will increase slowly. 
# # But, if the pattern is highly dispersed, then the F function will increase rapidly. 
# # The shape of this cumulative distribution is then compared to those constructed by calculating the same cumulative distribution between the random pattern and an additional, random one generated in each simulation step.



# # https://github.com/pysal/pointpats/blob/main/notebooks/distance_statistics-numpy-oriented.ipynb

# # Interevent Distance Functions

# # While both the F and G functions are useful, they only consider the distance between each point and its nearest point. 

# # the K function is a scaled version of the cumulative density function for all distances within a point pattern. 
# # As such, it's a "relative" of the function that considers all distances, not just the nearest neighbor distances.


# # we can see that the envelopes are generally above the observed function, meaining that our point pattern is dispersed.

# # The L function is a scaled version of function, defined in order to assist with interpretation. 
# # The expected value of the function increases with ; this makes sense, since the number of pairs of points closer than will increase as increases. 
# # So, we can define a normalization of that removes this increase as increases.




LR_std_dist=[]; 
LR_CLUSTERED = []; LR_DISPERSED = []
LR_GIO = []; LR_GIR = []
LR_FIO = []; LR_FIR = []
LR_GTEST = []; LR_FTEST = []
for time in times:
    tmp = LRwood_geotiffs_ds.wood.sel(time=time)
    for g in tqdm(LRbudget_reaches_redo):
        wood_c = tmp.rio.clip([g], tmp.rio.crs)
        label_img = label(wood_c==1)
        props = regionprops_table(label_img, properties=('area','axis_minor_length','centroid','orientation', 'coords'))
        # LR_props.append(props)
        try:
            dat = np.vstack(props['coords'])

            ## A measure of dispersion that is common in centrography is the standard distance. 
            # This measure provides the average distance away from the center of the point cloud (such as measured by the center of mass).
            # centrography.std_distance(coordinates)
            LR_std_dist.append(centrography.std_distance(dat))

            # print("Computing Ripley's G")
            g_test = distance_statistics.g_test(dat[::5], support=40, keep_simulations=True)
            LR_GTEST.append(g_test)

            ##where support=2
            # the observed % of nearest neighbor distances shorter than a distance of 2 
            g_index2_obs = g_test.statistic[np.where(g_test.support>2)[0][0]]
            g_index2_random = np.median(g_test.simulations, axis=0)[np.where(g_test.support>2)[0][0]]

            LR_GIO.append(g_index2_obs)
            LR_GIR.append(g_index2_random)
            LR_CLUSTERED.append(g_index2_obs>g_index2_random)

            # Where the G function works by analyzing the distance between points in the pattern, 
            # the F function works by analyzing the distance to points in the pattern from locations in empty space
            # it characterizes the typical distance from arbitrary points in empty space to the point pattern. 
            # More explicitly, the F accumulates, for a growing distance range, 
            # the percentage of points that can be found within that range from a random point pattern generated within the extent of the observed pattern. 
            # If the pattern has large gaps or empty areas, the F function will increase slowly. 
            # But, if the pattern is highly dispersed, then the F function will increase rapidly. 
            # The shape of this cumulative distribution is then compared to those constructed 
            # by calculating the same cumulative distribution between the random pattern and an additional, random one generated in each simulation step.

            # print("Computing Ripley's F")
            f_test = distance_statistics.f_test(dat[::5], support=40, keep_simulations=True)
            LR_FTEST.append(f_test)

            f_index2_obs = f_test.statistic[np.where(f_test.support>2)[0][0]]
            f_index2_random = np.median(f_test.simulations, axis=0)[np.where(f_test.support>2)[0][0]]

            LR_FIO.append(f_index2_obs)
            LR_FIR.append(f_index2_random)
            LR_DISPERSED.append(f_index2_obs>f_index2_random)
        except:

            LR_GIO.append(np.nan)
            LR_GIR.append(np.nan)
            LR_CLUSTERED.append(np.nan)

            LR_GTEST.append(np.nan)
            LR_FTEST.append(np.nan)

            LR_FIO.append(np.nan)
            LR_FIR.append(np.nan)
            LR_DISPERSED.append(np.nan)

np.savez('summaries/LR_spatial_metrics.npz', LR_GIO = LR_GIO, LR_GIR = LR_GIR, LR_CLUSTERED = LR_CLUSTERED, LR_GTEST = LR_GTEST, LR_FTEST=LR_FTEST, LR_FIO=LR_FIO, LR_FIR=LR_FIR, LR_DISPERSED=LR_DISPERSED)

#######################################################


MR_std_dist=[]; 
MR_CLUSTERED = []; MR_DISPERSED = []
MR_GIO = []; MR_GIR = []
MR_FIO = []; MR_FIR = []
MR_GTEST = []; MR_FTEST = []
for time in times:
    tmp = MRwood_geotiffs_ds.wood.sel(time=time)
    for g in tqdm(MRbudget_reaches_redo):
        wood_c = tmp.rio.clip([g], tmp.rio.crs)
        label_img = label(wood_c==1)
        props = regionprops_table(label_img, properties=('area','axis_minor_length','centroid','orientation', 'coords'))
        # MR_props.append(props)
        try:
            dat = np.vstack(props['coords'])

            ## A measure of dispersion that is common in centrography is the standard distance. 
            # This measure provides the average distance away from the center of the point cloud (such as measured by the center of mass).
            # centrography.std_distance(coordinates)
            MR_std_dist.append(centrography.std_distance(dat))

            # print("Computing Ripley's G")
            g_test = distance_statistics.g_test(dat[::5], support=40, keep_simulations=True)
            MR_GTEST.append(g_test)

            ##where support=2
            # the observed % of nearest neighbor distances shorter than a distance of 2 
            g_index2_obs = g_test.statistic[np.where(g_test.support>2)[0][0]]
            g_index2_random = np.median(g_test.simulations, axis=0)[np.where(g_test.support>2)[0][0]]

            MR_GIO.append(g_index2_obs)
            MR_GIR.append(g_index2_random)
            MR_CLUSTERED.append(g_index2_obs>g_index2_random)

            # Where the G function works by analyzing the distance between points in the pattern, 
            # the F function works by analyzing the distance to points in the pattern from locations in empty space
            # it characterizes the typical distance from arbitrary points in empty space to the point pattern. 
            # More explicitly, the F accumulates, for a growing distance range, 
            # the percentage of points that can be found within that range from a random point pattern generated within the extent of the observed pattern. 
            # If the pattern has large gaps or empty areas, the F function will increase slowly. 
            # But, if the pattern is highly dispersed, then the F function will increase rapidly. 
            # The shape of this cumulative distribution is then compared to those constructed 
            # by calculating the same cumulative distribution between the random pattern and an additional, random one generated in each simulation step.

            # print("Computing Ripley's F")
            f_test = distance_statistics.f_test(dat[::5], support=40, keep_simulations=True)
            MR_FTEST.append(f_test)

            f_index2_obs = f_test.statistic[np.where(f_test.support>2)[0][0]]
            f_index2_random = np.median(f_test.simulations, axis=0)[np.where(f_test.support>2)[0][0]]

            MR_FIO.append(f_index2_obs)
            MR_FIR.append(f_index2_random)
            MR_DISPERSED.append(f_index2_obs>f_index2_random)
        except:

            MR_GIO.append(np.nan)
            MR_GIR.append(np.nan)
            MR_CLUSTERED.append(np.nan)

            MR_GTEST.append(np.nan)
            MR_FTEST.append(np.nan)

            MR_FIO.append(np.nan)
            MR_FIR.append(np.nan)
            MR_DISPERSED.append(np.nan)



np.savez('summaries/MR_spatial_metrics.npz', MR_GIO = MR_GIO, MR_GIR = MR_GIR, MR_CLUSTERED = MR_CLUSTERED, MR_GTEST = MR_GTEST, MR_FTEST=MR_FTEST, MR_FIO=MR_FIO, MR_FIR=MR_FIR, MR_DISPERSED=MR_DISPERSED)





# plt.plot(y,xi,'ko', label='MR')

# plt.plot(y,xi,'rs', label='LR')
# plt.title('c) ', loc='left'); 
# plt.legend()
# plt.ylabel("Maximum\nautocorrelation (-)"); plt.xlabel("Maximum active\nchannel width (m)");
# plt.ylim(0,.4)




# fig, ax = plt.subplots(nrows=2, ncols=3)
# fig.set_size_inches(16,12)
# plt.subplots_adjust(wspace=0.6, hspace=0.2)

# # ####### flow
# # # plt.subplot(221)
# # # ax[0][0].plot(dt, np.nansum(MR_DISPERSED,axis=1)/len(MRbudget_reaches_redo),'k-', label='MR wood')
# # ax[0][0].plot(dt, np.nansum(LR_DISPERSED,axis=1)/len(LRbudget_reaches_redo),'r--', label='LR wood')
# # # ax[0][0].set_ylabel('Total wood area (m2)')

# # ax2 = ax[0][0].twinx()
# # # ax2.plot(dt, np.nansum(LR_DISPERSED,axis=1)/len(LRbudget_reaches_redo)); 
# # df_mnth_Q = sed_load.groupby(pd.PeriodIndex(sed_load['Day'], freq="M"))['Daily Discharge (m3/s)'].mean()
# # df_mnth_Q.plot(axes=ax2,label='Monthly mean\n discharge')
# # # ax2.plot(dt_sed[ind], sed_load['Daily Discharge (m3/s)'][ind],'b-', alpha=0.5, label='Daily discharge')
# # ax2.set_ylabel(r'Daily Discharge (m$^3$/s)', color='b')
# # ax[0][0].set_title('a) ', loc='left')

# # # ax2.plot(dt, OQ, 'b-o') #, legend='Discharge at image acquisition'
# # plt.show()

# plt.subplot(221)
# y = np.nansum(LR_DISPERSED,axis=1)/len(LRbudget_reaches_redo)
# plt.plot(dt, y/np.nanmax(y))

# # plt.subplot(222)
# df_mnth_Q = sed_load.groupby(pd.PeriodIndex(sed_load['Day'], freq="Y"))['Daily Discharge (m3/s)'].mean()
# df_mnth_Q = df_mnth_Q/np.nanmax(df_mnth_Q)
# df_mnth_Q.plot(label='Annual mean\n discharge')

# plt.subplot(222)
# y = np.nansum(LR_CLUSTERED,axis=1)/len(LRbudget_reaches_redo)
# plt.plot(dt, y/np.nanmax(y))

# # plt.subplot(223)
# df_mnth_L = sed_load.groupby(pd.PeriodIndex(sed_load['Day'], freq="Y"))['Total sediment discharge (tonnes)'].mean()
# df_mnth_L = df_mnth_L/np.nanmax(df_mnth_L)
# df_mnth_L.plot(label='Annual mean\n sediment load')

# # plt.subplot(223)

# # plt.ylabel(r"Water discharge (m/s$^3$)"); #plt.xlabel(r"Mean Autocorrelation coefficient")
# # plt.xlabel('')
# # plt.legend()
# # plt.title('c) ', loc='left');

# # plt.subplot(224)

# # plt.ylabel("Sediment load (tonnes)"); #plt.xlabel(r"Mean Autocorrelation coefficient")
# # plt.xlabel('')
# # plt.legend(loc=2)
# # plt.title('d)', loc='left');

# plt.show()


# plt.plot(np.nanmean(LR_FIO,axis=1)); plt.show()

# plt.plot(np.nansum(LR_CLUSTERED,axis=1), np.nansum(LR_DISPERSED,axis=1), 'ko'); plt.show()



# plt.subplot(337)
# plt.plot(OQ[1:], np.median(MR_PC2,axis=1), 'ko', label='MR')
# plt.plot(OQ[1:], np.median(LR_PC2,axis=1), 'rs', label='LR')
# plt.title('e) ', loc='left'); 
# plt.legend()
# plt.xlabel("Discharge (m$^3$/s)"); plt.ylabel("Median autocorrelation (-)");
# # plt.ylim(0,.16)

# plt.subplot(338)
# plt.plot(np.log(OS[1:]), np.median(MR_PC2,axis=1), 'ko', label='MR')
# plt.plot(np.log(OS[1:]), np.median(LR_PC2,axis=1), 'rs', label='LR')
# plt.title('f) ', loc='left'); 
# plt.legend()
# plt.xlabel("Sediment load (log tonnes)"); plt.ylabel("Median autocorrelation (-)");
# # plt.ylim(0,.16)

# plt.subplot(339)
# x = np.max(MR_PC2g,axis=0)
# # x = np.max(MR_PC2g,axis=0)
# y = MR_full_width.copy()
# xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x)
# plt.plot(y,xi,'ko', label='MR')
# x = np.max(LR_PC2g,axis=0)
# # x = np.median(LR_PC2g,axis=0)
# y = LR_full_width.copy()
# xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x)
# plt.plot(y,xi,'rs', label='LR')
# plt.title('g) ', loc='left'); 
# plt.legend()
# plt.ylabel("Max\nautocorrelation (-)"); plt.xlabel("Max channel\n width (m)");
# # plt.ylim(0,.66)


# MR_IC2 = inpaint_nans(np.flipud(MR_IC))
# LR_IC2 = inpaint_nans(np.flipud(LR_IC))

# MR_IC2[MR_IC2<0] = 0
# LR_IC2[LR_IC2<0] = 0



# ########################################
# plt.figure(figsize=(16,12))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(231)
# plt.imshow(MR_IC2, cmap='inferno', extent=[MR[0], MR[-1] , dt[1], dt[-1]], aspect='auto', vmax=1, vmin=0)
# # cb=plt.colorbar(); cb.set_label(r"Correlation coefficient")
# plt.gca().invert_yaxis()
# # plt.gca().invert_xaxis()
# plt.title('a) ', loc='left'); plt.xlabel("River kilometer"); 

# plt.subplot(232)
# plt.imshow(LR_IC2, cmap='inferno', extent=[LR[0], LR[-1] , dt[1], dt[-1]], aspect='auto', vmax=1, vmin=0)
# # cb=plt.colorbar(); cb.set_label(r"Correlation coefficient")
# plt.gca().invert_yaxis()
# # plt.gca().invert_xaxis()
# # plt.title('b) ', loc='left'); plt.xlabel("River kilometer"); 

# plt.subplot(233)
# plt.plot(np.nanmean(MR_IC2,axis=1),dt[1:],'k-', label='MR')
# im=plt.plot(np.nanmean(LR_IC2,axis=1),dt[1:],'r--', label='LR')
# plt.legend(fontsize=8)
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel(r"Mean autocorrelation");
# plt.xlim(0,.8)
# # plt.colorbar(im)

# plt.subplot(234)
# im2=plt.plot(MR,np.nanmean(MR_IC2,axis=0),'k',lw=2)
# # cb=plt.colorbar();
# # plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.title('c) ', loc='left'); plt.xlabel("River kilometer"); 
# plt.ylim(0,.8)
# plt.xlim(20,12)

# plt.subplot(235)
# im3=plt.plot(LR,np.nanmean(LR_IC2,axis=0),'r--',lw=2)
# # cb=plt.colorbar();
# # plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# # plt.title('c) ', loc='left'); plt.xlabel("River kilometer"); 
# plt.ylim(0,.8)
# plt.xlim(11,2)

# plt.subplot(236)
# x = np.max(MR_IC2,axis=0)
# y = MR_widths.copy()
# xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x)
# plt.plot(y,xi,'ko', label='MR')
# x = np.max(LR_IC2,axis=0)
# y = LR_widths.copy()
# xi = np.interp(np.linspace(np.nanmin(x), np.nanmax(x),len(y)),np.linspace(0,len(x),len(x)),x)
# plt.plot(y,xi,'rs', label='LR')
# plt.title('c) ', loc='left'); 
# plt.legend()
# plt.ylabel("Maximum\nautocorrelation (-)"); plt.xlabel("Maximum active\nchannel width (m)");
# plt.ylim(0,.8)



# # plt.show()
# plt.savefig("summaries/wood_spacetime_persistence_pairwiseIC.png", dpi=300, bbox_inches="tight")
# plt.close()




# plt.subplot(323)
# plt.plot(MR,np.nanmean(MR_PC,axis=0),'k--', label='MR')

# plt.subplot(324)
# plt.plot(LR,np.nanmean(LR_PC,axis=0),'r--', label='LR')
# # plt.title('f) ', loc='left'); plt.ylabel(r"Mean autocorrelation");
# plt.title('f) ', loc='left'); plt.ylabel(r"Mean autocorrelation"); plt.xlabel("River kilometer"); 



# plt.plot(dt[1:],np.nanmean(LR_PC,axis=1),'r--', label='LR, co-location')
# plt.legend(fontsize=8)
# plt.ylim(0,.7)



# plt.subplot(332)
# plt.imshow(np.flipud(MR_PC), cmap='inferno', extent=[MR[0], MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Pearson correlation coefficient")
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(322)
# plt.imshow(np.flipud(MR_MC), cmap='inferno', extent=[MR[0], MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Manders co-location coefficient")
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel("River kilometer"); 

# plt.subplot(323)
# plt.imshow(np.flipud(LR_IC), cmap='inferno', extent=[MR[0], LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient")
# plt.gca().invert_yaxis()
# plt.title('c) ', loc='left'); plt.xlabel("River kilometer"); 

# plt.subplot(335)
# plt.imshow(np.flipud(LR_PC), cmap='inferno', extent=[0, LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Pearson correlation coefficient")
# plt.gca().invert_yaxis()
# plt.title('e) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(324)
# plt.imshow(np.flipud(LR_MC), cmap='inferno', extent=[MR[0], LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Manders co-location coefficient")
# plt.gca().invert_yaxis()
# plt.title('d) ', loc='left'); plt.xlabel("River kilometer"); 

# plt.subplot(337)
# plt.plot(MR, np.nanmean(MR_PC,axis=0),'k-', label='MR')
# plt.plot(LR, np.nanmean(LR_PC,axis=0),'r--', label='LR')
# plt.ylabel(r"Mean autocorrelation"); plt.xlabel("River kilometer(km)"); 
# plt.legend()
# plt.title('g) ', loc='left')

# plt.subplot(325)
# plt.plot(dt[1:],np.nanmean(MR_IC,axis=1),'k-', label='MR, intersection')
# plt.plot(dt[1:],np.nanmean(LR_IC,axis=1),'r-', label='LR, intersection')
# plt.title('e) ', loc='left'); plt.ylabel(r"Mean persistence");

# plt.plot(dt[1:],np.nanmean(MR_PC,axis=1),'k--', label='MR, co-location')
# plt.plot(dt[1:],np.nanmean(LR_PC,axis=1),'r--', label='LR, co-location')
# plt.legend(fontsize=8)

# plt.subplot(326)
# plt.plot(MR,np.nanmean(MR_IC,axis=0),'k-', label='MR, intersection')
# plt.plot(LR,np.nanmean(LR_IC,axis=0),'r-', label='LR, intersection')

# plt.plot(MR,np.nanmean(MR_PC,axis=0),'k--', label='MR, co-location')
# plt.plot(LR,np.nanmean(LR_PC,axis=0),'r--', label='LR, co-location')
# # plt.title('f) ', loc='left'); plt.ylabel(r"Mean autocorrelation");
# plt.legend(fontsize=8)
# plt.title('f) ', loc='left'); plt.ylabel(r"Mean persistence"); plt.xlabel("River kilometer"); 





# ########################################

# ### per elevation bin
# #### MR
# wood0 = MRwood_geotiffs_ds.wood.sel(time=times[0])
        
# MR_IC1 = []; MR_PC1 = []; MR_PV1=[]
# MR_IC2 = []; MR_PC2 = []; MR_PV2=[]
# MR_IC3 = []; MR_PC3 = []; MR_PV3=[]
# MR_IC4 = []; MR_PC4 = []; MR_PV4=[]
# MR_IC5 = []; MR_PC5 = []; MR_PV5=[]
# MR_IC6 = []; MR_PC6 = []; MR_PV6=[]
# MR_IC7 = []; MR_PC7 = []; MR_PV7=[]
# MR_IC8 = []; MR_PC8 = []; MR_PV8=[]
# MR_IC9 = []; MR_PC9 = []; MR_PV9=[]
# MR_IC10 = []; MR_PC10 = []; MR_PV10=[]
# MR_IC11 = []; MR_PC11 = []; MR_PV11=[]
# MR_IC12 = []; MR_PC12 = []; MR_PV12=[]
# MR_IC13 = []; MR_PC13 = []; MR_PV13=[]
# MR_IC14 = []; MR_PC14 = []; MR_PV14=[]

# for time in times[1:]:
#     print(time)
#     tmp = MRwood_geotiffs_ds.wood.sel(time=time)
#     dem_tmp = MR_dem_detrend_geotiffs_ds.dem.sel(time=time)
#     dem_min = np.nanmin(dem_tmp)
#     if dem_min<1:
#         dem_tmp = dem_tmp-dem_min

#     for g in MRbudget_reaches_redo:
#         wood_c = tmp.rio.clip([g], tmp.rio.crs)
#         wood_c0 = wood0.rio.clip([g], tmp.rio.crs)
#         dem_c = dem_tmp.rio.clip([g], dem_tmp.rio.crs).to_numpy()

#         bin1 = wood_c*((dem_c < 1))
#         bin2 = wood_c*((dem_c >= 1) & (dem_c < 2))
#         bin3 = wood_c*((dem_c >= 2) & (dem_c < 3))
#         bin4 = wood_c*((dem_c >= 3) & (dem_c < 4))
#         bin5 = wood_c*((dem_c >= 4) & (dem_c < 5))
#         bin6 = wood_c*((dem_c >= 5) & (dem_c < 6))
#         bin7 = wood_c*((dem_c >= 7) & (dem_c < 8))
#         bin8 = wood_c*((dem_c >= 8) & (dem_c < 9))
#         bin9 = wood_c*((dem_c >= 9) & (dem_c < 10))
#         bin10 = wood_c*((dem_c >= 10) & (dem_c < 11))
#         bin11 = wood_c*((dem_c >= 11) & (dem_c < 12))
#         bin12 = wood_c*((dem_c >= 12) & (dem_c < 13))
#         bin13 = wood_c*((dem_c >= 13) & (dem_c < 14))
#         bin14 = wood_c*((dem_c > 14))

#         bin1_0 = wood_c0*((dem_c < 1))
#         bin2_0 = wood_c0*((dem_c >= 2) & (dem_c < 2))
#         bin3_0 = wood_c0*((dem_c >= 3) & (dem_c < 3))
#         bin4_0 = wood_c0*((dem_c >= 4) & (dem_c < 4))
#         bin5_0 = wood_c0*((dem_c >= 5) & (dem_c < 5))
#         bin6_0 = wood_c0*((dem_c >= 6) & (dem_c < 6))
#         bin7_0 = wood_c0*((dem_c >= 7) & (dem_c < 7))
#         bin8_0 = wood_c0*((dem_c >= 8) & (dem_c < 9))
#         bin9_0 = wood_c0*((dem_c >= 9) & (dem_c < 10))
#         bin10_0 = wood_c0*((dem_c >= 10) & (dem_c < 11))
#         bin11_0 = wood_c0*((dem_c >= 11) & (dem_c < 12))
#         bin12_0 = wood_c0*((dem_c >= 12) & (dem_c < 13))
#         bin13_0 = wood_c0*((dem_c >= 13) & (dem_c < 14))
#         bin14_0 = wood_c0*((dem_c > 14))

#         MR_IC1.append(intersection_coeff(bin1_0.to_numpy()==1, bin1.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin1_0.to_numpy()==1, bin1.to_numpy()==1)
#         MR_PC1.append(pcc)
#         MR_PV1.append(pval)

#         MR_IC2.append(intersection_coeff(bin2_0.to_numpy()==1, bin2.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin2_0.to_numpy()==1, bin2.to_numpy()==1)
#         MR_PC2.append(pcc)
#         MR_PV2.append(pval)

#         MR_IC3.append(intersection_coeff(bin3_0.to_numpy()==1, bin3.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin3_0.to_numpy()==1, bin3.to_numpy()==1)
#         MR_PC3.append(pcc)
#         MR_PV3.append(pval)

#         MR_IC4.append(intersection_coeff(bin4_0.to_numpy()==1, bin4.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin4_0.to_numpy()==1, bin4.to_numpy()==1)
#         MR_PC4.append(pcc)
#         MR_PV4.append(pval)

#         MR_IC5.append(intersection_coeff(bin5_0.to_numpy()==1, bin5.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin5_0.to_numpy()==1, bin5.to_numpy()==1)
#         MR_PC5.append(pcc)
#         MR_PV5.append(pval)

#         MR_IC6.append(intersection_coeff(bin6_0.to_numpy()==1, bin6.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin6_0.to_numpy()==1, bin6.to_numpy()==1)
#         MR_PC6.append(pcc)
#         MR_PV6.append(pval)

#         MR_IC7.append(intersection_coeff(bin7_0.to_numpy()==1, bin7.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin7_0.to_numpy()==1, bin7.to_numpy()==1)
#         MR_PC7.append(pcc)
#         MR_PV7.append(pval)

#         MR_IC8.append(intersection_coeff(bin8_0.to_numpy()==1, bin8.to_numpy()==1))
#         # MR_MO8.append(manders_overlap_coeff(bin8_0.to_numpy()==1, bin8.to_numpy()==1))
#         # MR_MC8.append(manders_coloc_coeff(bin8_0.to_numpy()==1, bin8.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin8_0.to_numpy()==1, bin8.to_numpy()==1)
#         MR_PC8.append(pcc)
#         MR_PV8.append(pval)

#         MR_IC9.append(intersection_coeff(bin9_0.to_numpy()==1, bin9.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin9_0.to_numpy()==1, bin9.to_numpy()==1)
#         MR_PC9.append(pcc)
#         MR_PV9.append(pval)

#         MR_IC10.append(intersection_coeff(bin10_0.to_numpy()==1, bin10.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin10_0.to_numpy()==1, bin10.to_numpy()==1)
#         MR_PC10.append(pcc)
#         MR_PV10.append(pval)

#         MR_IC11.append(intersection_coeff(bin11_0.to_numpy()==1, bin11.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin11_0.to_numpy()==1, bin11.to_numpy()==1)
#         MR_PC11.append(pcc)
#         MR_PV11.append(pval)        

#         MR_IC12.append(intersection_coeff(bin12_0.to_numpy()==1, bin12.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin12_0.to_numpy()==1, bin12.to_numpy()==1)
#         MR_PC12.append(pcc)
#         MR_PV12.append(pval)

#         MR_IC13.append(intersection_coeff(bin13_0.to_numpy()==1, bin13.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin13_0.to_numpy()==1, bin13.to_numpy()==1)
#         MR_PC13.append(pcc)
#         MR_PV13.append(pval)

#         MR_IC14.append(intersection_coeff(bin14_0.to_numpy()==1, bin14.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin14_0.to_numpy()==1, bin14.to_numpy()==1)
#         MR_PC14.append(pcc)
#         MR_PV14.append(pval)


# np.savez('summaries/MR_association_metrics_heightbins.npz', MR_IC1 = MR_IC1, MR_PC1 = MR_PC1, MR_PV1=MR_PV1, MR_IC2=MR_IC2, MR_PC2=MR_PC2, MR_PV2=MR_PV2, MR_IC3=MR_IC3, MR_PC3=MR_PC3, MR_PV3=MR_PV3, MR_IC4=MR_IC4, MR_PC4=MR_PC4, MR_PV4=MR_PV4, MR_IC5=MR_IC5, MR_PC5=MR_PC5, MR_PV5=MR_PV5, MR_IC6=MR_IC6, MR_PC6=MR_PC6, MR_PV6=MR_PV6, MR_IC7=MR_IC7, MR_PC7=MR_PC7, MR_PV7=MR_PV7, MR_IC8=MR_IC8, MR_PC8=MR_PC8, MR_PV8=MR_PV8, MR_IC9=MR_IC9, MR_PC9=MR_PC9, MR_PV9=MR_PV9,MR_IC10=MR_IC10, MR_PC10=MR_PC10, MR_PV10=MR_PV10,MR_IC11=MR_IC11, MR_PC11=MR_PC11, MR_PV11=MR_PV11, MR_IC12=MR_IC12, MR_PC12=MR_PC12, MR_PV12=MR_PV12,MR_IC13=MR_IC13, MR_PC13=MR_PC13, MR_PV13=MR_PV13,MR_IC14=MR_IC14, MR_PC14=MR_PC14, MR_PV14=MR_PV14)



# with np.load('summaries/MR_association_metrics_heightbins.npz', allow_pickle=True) as f:
#     MR_IC1= f['MR_IC1']; MR_PC1= f['MR_PC1']; MR_PV1= f['MR_PV1']; 
#     MR_IC2= f['MR_IC2']; MR_PC2= f['MR_PC2']; MR_PV2= f['MR_PV2']; 
#     MR_IC3= f['MR_IC3']; MR_PC3= f['MR_PC3']; MR_PV3= f['MR_PV3']
#     MR_IC4= f['MR_IC4']; MR_PC4= f['MR_PC4']; MR_PV4= f['MR_PV4']; 
#     MR_IC5= f['MR_IC5']; MR_PC5= f['MR_PC5']; MR_PV5= f['MR_PV5']; 
#     MR_IC6= f['MR_IC6']; MR_PC6= f['MR_PC6']; MR_PV6= f['MR_PV6']; 
#     MR_IC7= f['MR_IC7']; MR_PC7= f['MR_PC7']; MR_PV7= f['MR_PV7']; 
#     MR_IC8= f['MR_IC8']; MR_PC8= f['MR_PC8']; MR_PV8= f['MR_PV8']
#     MR_IC9= f['MR_IC9']; MR_PC9= f['MR_PC9']; MR_PV9= f['MR_PV9']
#     MR_IC10= f['MR_IC10']; MR_PC10= f['MR_PC10']; MR_PV10= f['MR_PV10']
#     MR_IC11= f['MR_IC11']; MR_PC11= f['MR_PC11']; MR_PV11= f['MR_PV11']
#     MR_IC12= f['MR_IC12']; MR_PC12= f['MR_PC12']; MR_PV12= f['MR_PV12']
#     MR_IC13= f['MR_IC13']; MR_PC13= f['MR_PC13']; MR_PV13= f['MR_PV13']
#     MR_IC14= f['MR_IC14']; MR_PC14= f['MR_PC14']; MR_PV14= f['MR_PV14']


# MR_IC1 = np.array(MR_IC1).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC2 = np.array(MR_IC2).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC3 = np.array(MR_IC3).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC4 = np.array(MR_IC4).reshape(len(times)-1,len(MRbudget_reaches_redo)) 
# MR_IC5 = np.array(MR_IC5).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC6 = np.array(MR_IC6).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC7 = np.array(MR_IC7).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC8 = np.array(MR_IC8).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC9 = np.array(MR_IC9).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC10 = np.array(MR_IC10).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC11 = np.array(MR_IC11).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC12 = np.array(MR_IC12).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC13 = np.array(MR_IC13).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_IC14 = np.array(MR_IC14).reshape(len(times)-1,len(MRbudget_reaches_redo))


# MR_PC1 = np.array(MR_PC1).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC2 = np.array(MR_PC2).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC3 = np.array(MR_PC3).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC4 = np.array(MR_PC4).reshape(len(times)-1,len(MRbudget_reaches_redo)) 
# MR_PC5 = np.array(MR_PC5).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC6 = np.array(MR_PC6).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC7 = np.array(MR_PC7).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC8 = np.array(MR_PC8).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC9 = np.array(MR_PC9).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC10 = np.array(MR_PC10).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC11 = np.array(MR_PC11).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC12 = np.array(MR_PC12).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC13 = np.array(MR_PC13).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PC14 = np.array(MR_PC14).reshape(len(times)-1,len(MRbudget_reaches_redo))


# MR_PV1 = np.array(MR_PV1).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV2 = np.array(MR_PV2).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV3 = np.array(MR_PV3).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV4 = np.array(MR_PV4).reshape(len(times)-1,len(MRbudget_reaches_redo)) 
# MR_PV5 = np.array(MR_PV5).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV6 = np.array(MR_PV6).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV7 = np.array(MR_PV7).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV8 = np.array(MR_PV8).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV9 = np.array(MR_PV9).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV10 = np.array(MR_PV10).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV11 = np.array(MR_PV11).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV12 = np.array(MR_PV12).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV13 = np.array(MR_PV13).reshape(len(times)-1,len(MRbudget_reaches_redo))
# MR_PV14 = np.array(MR_PV14).reshape(len(times)-1,len(MRbudget_reaches_redo))



# ######################################################################################
# #### LR
# wood0 = LRwood_geotiffs_ds.wood.sel(time=times[0])
        
# LR_IC1 = []; LR_PC1 = []; LR_PV1=[]
# LR_IC2 = []; LR_PC2 = []; LR_PV2=[]
# LR_IC3 = []; LR_PC3 = []; LR_PV3=[]
# LR_IC4 = []; LR_PC4 = []; LR_PV4=[]
# LR_IC5 = []; LR_PC5 = []; LR_PV5=[]
# LR_IC6 = []; LR_PC6 = []; LR_PV6=[]
# LR_IC7 = []; LR_PC7 = []; LR_PV7=[]
# LR_IC8 = []; LR_PC8 = []; LR_PV8=[]
# LR_IC9 = []; LR_PC9 = []; LR_PV9=[]
# LR_IC10 = []; LR_PC10 = []; LR_PV10=[]
# LR_IC11 = []; LR_PC11 = []; LR_PV11=[]
# LR_IC12 = []; LR_PC12 = []; LR_PV12=[]
# LR_IC13 = []; LR_PC13 = []; LR_PV13=[]
# LR_IC14 = []; LR_PC14 = []; LR_PV14=[]

# for time in times[1:]:
#     print(time)
#     tmp = LRwood_geotiffs_ds.wood.sel(time=time)
#     dem_tmp = LR_dem_detrend_geotiffs_ds.dem.sel(time=time)
#     dem_min = np.nanmin(dem_tmp)
#     if dem_min<1:
#         dem_tmp = dem_tmp-dem_min

#     for g in LRbudget_reaches_redo:
#         wood_c = tmp.rio.clip([g], tmp.rio.crs)
#         wood_c0 = wood0.rio.clip([g], tmp.rio.crs)
#         dem_c = dem_tmp.rio.clip([g], dem_tmp.rio.crs).to_numpy()

#         bin1 = wood_c*((dem_c < 1))
#         bin2 = wood_c*((dem_c >= 1) & (dem_c < 2))
#         bin3 = wood_c*((dem_c >= 2) & (dem_c < 3))
#         bin4 = wood_c*((dem_c >= 3) & (dem_c < 4))
#         bin5 = wood_c*((dem_c >= 4) & (dem_c < 5))
#         bin6 = wood_c*((dem_c >= 5) & (dem_c < 6))
#         bin7 = wood_c*((dem_c >= 7) & (dem_c < 8))
#         bin8 = wood_c*((dem_c >= 8) & (dem_c < 9))
#         bin9 = wood_c*((dem_c >= 9) & (dem_c < 10))
#         bin10 = wood_c*((dem_c >= 10) & (dem_c < 11))
#         bin11 = wood_c*((dem_c >= 11) & (dem_c < 12))
#         bin12 = wood_c*((dem_c >= 12) & (dem_c < 13))
#         bin13 = wood_c*((dem_c >= 13) & (dem_c < 14))
#         bin14 = wood_c*((dem_c > 14))

#         bin1_0 = wood_c0*((dem_c < 1))
#         bin2_0 = wood_c0*((dem_c >= 2) & (dem_c < 2))
#         bin3_0 = wood_c0*((dem_c >= 3) & (dem_c < 3))
#         bin4_0 = wood_c0*((dem_c >= 4) & (dem_c < 4))
#         bin5_0 = wood_c0*((dem_c >= 5) & (dem_c < 5))
#         bin6_0 = wood_c0*((dem_c >= 6) & (dem_c < 6))
#         bin7_0 = wood_c0*((dem_c >= 7) & (dem_c < 7))
#         bin8_0 = wood_c0*((dem_c >= 8) & (dem_c < 9))
#         bin9_0 = wood_c0*((dem_c >= 9) & (dem_c < 10))
#         bin10_0 = wood_c0*((dem_c >= 10) & (dem_c < 11))
#         bin11_0 = wood_c0*((dem_c >= 11) & (dem_c < 12))
#         bin12_0 = wood_c0*((dem_c >= 12) & (dem_c < 13))
#         bin13_0 = wood_c0*((dem_c >= 13) & (dem_c < 14))
#         bin14_0 = wood_c0*((dem_c > 14))

#         LR_IC1.append(intersection_coeff(bin1_0.to_numpy()==1, bin1.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin1_0.to_numpy()==1, bin1.to_numpy()==1)
#         LR_PC1.append(pcc)
#         LR_PV1.append(pval)

#         LR_IC2.append(intersection_coeff(bin2_0.to_numpy()==1, bin2.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin2_0.to_numpy()==1, bin2.to_numpy()==1)
#         LR_PC2.append(pcc)
#         LR_PV2.append(pval)

#         LR_IC3.append(intersection_coeff(bin3_0.to_numpy()==1, bin3.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin3_0.to_numpy()==1, bin3.to_numpy()==1)
#         LR_PC3.append(pcc)
#         LR_PV3.append(pval)

#         LR_IC4.append(intersection_coeff(bin4_0.to_numpy()==1, bin4.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin4_0.to_numpy()==1, bin4.to_numpy()==1)
#         LR_PC4.append(pcc)
#         LR_PV4.append(pval)

#         LR_IC5.append(intersection_coeff(bin5_0.to_numpy()==1, bin5.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin5_0.to_numpy()==1, bin5.to_numpy()==1)
#         LR_PC5.append(pcc)
#         LR_PV5.append(pval)

#         LR_IC6.append(intersection_coeff(bin6_0.to_numpy()==1, bin6.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin6_0.to_numpy()==1, bin6.to_numpy()==1)
#         LR_PC6.append(pcc)
#         LR_PV6.append(pval)

#         LR_IC7.append(intersection_coeff(bin7_0.to_numpy()==1, bin7.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin7_0.to_numpy()==1, bin7.to_numpy()==1)
#         LR_PC7.append(pcc)
#         LR_PV7.append(pval)

#         LR_IC8.append(intersection_coeff(bin8_0.to_numpy()==1, bin8.to_numpy()==1))
#         # LR_MO8.append(manders_overlap_coeff(bin8_0.to_numpy()==1, bin8.to_numpy()==1))
#         # LR_MC8.append(manders_coloc_coeff(bin8_0.to_numpy()==1, bin8.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin8_0.to_numpy()==1, bin8.to_numpy()==1)
#         LR_PC8.append(pcc)
#         LR_PV8.append(pval)

#         LR_IC9.append(intersection_coeff(bin9_0.to_numpy()==1, bin9.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin9_0.to_numpy()==1, bin9.to_numpy()==1)
#         LR_PC9.append(pcc)
#         LR_PV9.append(pval)

#         LR_IC10.append(intersection_coeff(bin10_0.to_numpy()==1, bin10.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin10_0.to_numpy()==1, bin10.to_numpy()==1)
#         LR_PC10.append(pcc)
#         LR_PV10.append(pval)

#         LR_IC11.append(intersection_coeff(bin11_0.to_numpy()==1, bin11.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin11_0.to_numpy()==1, bin11.to_numpy()==1)
#         LR_PC11.append(pcc)
#         LR_PV11.append(pval)        

#         LR_IC12.append(intersection_coeff(bin12_0.to_numpy()==1, bin12.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin12_0.to_numpy()==1, bin12.to_numpy()==1)
#         LR_PC12.append(pcc)
#         LR_PV12.append(pval)

#         LR_IC13.append(intersection_coeff(bin13_0.to_numpy()==1, bin13.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin13_0.to_numpy()==1, bin13.to_numpy()==1)
#         LR_PC13.append(pcc)
#         LR_PV13.append(pval)

#         LR_IC14.append(intersection_coeff(bin14_0.to_numpy()==1, bin14.to_numpy()==1))
#         pcc, pval = pearson_corr_coeff(bin14_0.to_numpy()==1, bin14.to_numpy()==1)
#         LR_PC14.append(pcc)
#         LR_PV14.append(pval)


# np.savez('summaries/LR_association_metrics_heightbins.npz', LR_IC1 = LR_IC1, LR_PC1 = LR_PC1, LR_PV1=LR_PV1, LR_IC2=LR_IC2, LR_PC2=LR_PC2, LR_PV2=LR_PV2, LR_IC3=LR_IC3, LR_PC3=LR_PC3, LR_PV3=LR_PV3, LR_IC4=LR_IC4, LR_PC4=LR_PC4, LR_PV4=LR_PV4, LR_IC5=LR_IC5, LR_PC5=LR_PC5, LR_PV5=LR_PV5, LR_IC6=LR_IC6, LR_PC6=LR_PC6, LR_PV6=LR_PV6, LR_IC7=LR_IC7, LR_PC7=LR_PC7, LR_PV7=LR_PV7, LR_IC8=LR_IC8, LR_PC8=LR_PC8, LR_PV8=LR_PV8, LR_IC9=LR_IC9, LR_PC9=LR_PC9, LR_PV9=LR_PV9,LR_IC10=LR_IC10, LR_PC10=LR_PC10, LR_PV10=LR_PV10,LR_IC11=LR_IC11, LR_PC11=LR_PC11, LR_PV11=LR_PV11, LR_IC12=LR_IC12, LR_PC12=LR_PC12, LR_PV12=LR_PV12,LR_IC13=LR_IC13, LR_PC13=LR_PC13, LR_PV13=LR_PV13,LR_IC14=LR_IC14, LR_PC14=LR_PC14, LR_PV14=LR_PV14)


# with np.load('summaries/LR_association_metrics_heightbins.npz', allow_pickle=True) as f:
#     LR_IC1= f['LR_IC1']; LR_PC1= f['LR_PC1']; LR_PV1= f['LR_PV1']; 
#     LR_IC2= f['LR_IC2']; LR_PC2= f['LR_PC2']; LR_PV2= f['LR_PV2']; 
#     LR_IC3= f['LR_IC3']; LR_PC3= f['LR_PC3']; LR_PV3= f['LR_PV3']
#     LR_IC4= f['LR_IC4']; LR_PC4= f['LR_PC4']; LR_PV4= f['LR_PV4']; 
#     LR_IC5= f['LR_IC5']; LR_PC5= f['LR_PC5']; LR_PV5= f['LR_PV5']; 
#     LR_IC6= f['LR_IC6']; LR_PC6= f['LR_PC6']; LR_PV6= f['LR_PV6']; 
#     LR_IC7= f['LR_IC7']; LR_PC7= f['LR_PC7']; LR_PV7= f['LR_PV7']; 
#     LR_IC8= f['LR_IC8']; LR_PC8= f['LR_PC8']; LR_PV8= f['LR_PV8']
#     LR_IC9= f['LR_IC9']; LR_PC9= f['LR_PC9']; LR_PV9= f['LR_PV9']
#     LR_IC10= f['LR_IC10']; LR_PC10= f['LR_PC10']; LR_PV10= f['LR_PV10']
#     LR_IC11= f['LR_IC11']; LR_PC11= f['LR_PC11']; LR_PV11= f['LR_PV11']
#     LR_IC12= f['LR_IC12']; LR_PC12= f['LR_PC12']; LR_PV12= f['LR_PV12']
#     LR_IC13= f['LR_IC13']; LR_PC13= f['LR_PC13']; LR_PV13= f['LR_PV13']
#     LR_IC14= f['LR_IC14']; LR_PC14= f['LR_PC14']; LR_PV14= f['LR_PV14']


# LR_IC1 = np.array(LR_IC1).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC2 = np.array(LR_IC2).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC3 = np.array(LR_IC3).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC4 = np.array(LR_IC4).reshape(len(times)-1,len(LRbudget_reaches_redo)) 
# LR_IC5 = np.array(LR_IC5).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC6 = np.array(LR_IC6).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC7 = np.array(LR_IC7).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC8 = np.array(LR_IC8).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC9 = np.array(LR_IC9).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC10 = np.array(LR_IC10).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC11 = np.array(LR_IC11).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC12 = np.array(LR_IC12).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC13 = np.array(LR_IC13).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_IC14 = np.array(LR_IC14).reshape(len(times)-1,len(LRbudget_reaches_redo))


# LR_PC1 = np.array(LR_PC1).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC2 = np.array(LR_PC2).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC3 = np.array(LR_PC3).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC4 = np.array(LR_PC4).reshape(len(times)-1,len(LRbudget_reaches_redo)) 
# LR_PC5 = np.array(LR_PC5).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC6 = np.array(LR_PC6).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC7 = np.array(LR_PC7).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC8 = np.array(LR_PC8).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC9 = np.array(LR_PC9).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC10 = np.array(LR_PC10).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC11 = np.array(LR_PC11).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC12 = np.array(LR_PC12).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC13 = np.array(LR_PC13).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PC14 = np.array(LR_PC14).reshape(len(times)-1,len(LRbudget_reaches_redo))


# LR_PV1 = np.array(LR_PV1).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV2 = np.array(LR_PV2).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV3 = np.array(LR_PV3).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV4 = np.array(LR_PV4).reshape(len(times)-1,len(LRbudget_reaches_redo)) 
# LR_PV5 = np.array(LR_PV5).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV6 = np.array(LR_PV6).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV7 = np.array(LR_PV7).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV8 = np.array(LR_PV8).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV9 = np.array(LR_PV9).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV10 = np.array(LR_PV10).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV11 = np.array(LR_PV11).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV12 = np.array(LR_PV12).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV13 = np.array(LR_PV13).reshape(len(times)-1,len(LRbudget_reaches_redo))
# LR_PV14 = np.array(LR_PV14).reshape(len(times)-1,len(LRbudget_reaches_redo))


# ## spatial averages

# imMR = np.vstack((np.nanmean(MR_PC1,axis=0),np.nanmean(MR_PC2,axis=0),np.nanmean(MR_PC3,axis=0),np.nanmean(MR_PC4,axis=0),np.nanmean(MR_PC5,axis=0),np.nanmean(MR_PC6,axis=0),np.nanmean(MR_PC7,axis=0),np.nanmean(MR_PC8,axis=0),np.nanmean(MR_PC9,axis=0),np.nanmean(MR_PC10,axis=0),np.nanmean(MR_PC11,axis=0),np.nanmean(MR_PC12,axis=0),np.nanmean(MR_PC13,axis=0),np.nanmean(MR_PC14,axis=0)))

# imLR = np.vstack((np.nanmean(LR_PC1,axis=1),np.nanmean(LR_PC2,axis=1),np.nanmean(LR_PC3,axis=1),np.nanmean(LR_PC4,axis=1),np.nanmean(LR_PC5,axis=1),np.nanmean(LR_PC6,axis=1),np.nanmean(LR_PC7,axis=1),np.nanmean(LR_PC8,axis=1),np.nanmean(LR_PC9,axis=1),np.nanmean(LR_PC10,axis=1),np.nanmean(LR_PC11,axis=1),np.nanmean(LR_PC12,axis=1),np.nanmean(LR_PC13,axis=1),np.nanmean(LR_PC14,axis=1)))





# imMR = np.vstack((np.mean(MR_IC3,axis=1),np.mean(MR_IC4,axis=1),np.mean(MR_IC5,axis=1),np.mean(MR_IC6,axis=1),np.mean(MR_IC7,axis=1),np.mean(MR_IC8,axis=1),np.mean(MR_IC9,axis=1),np.mean(MR_IC10,axis=1),np.mean(MR_IC11,axis=1),np.mean(MR_IC12,axis=1),np.mean(MR_IC13,axis=1),np.mean(MR_IC14,axis=1)))

# imLR = np.vstack((np.mean(LR_IC3,axis=1),np.mean(LR_IC4,axis=1),np.mean(LR_IC5,axis=1),np.mean(LR_IC6,axis=1),np.mean(LR_IC7,axis=1),np.mean(LR_IC8,axis=1),np.mean(LR_IC9,axis=1),np.mean(LR_IC10,axis=1),np.mean(LR_IC11,axis=1),np.mean(LR_IC12,axis=1),np.mean(LR_IC13,axis=1),np.mean(LR_IC14,axis=1)))


# imMR = np.vstack((np.mean(MR_PC3,axis=1),np.mean(MR_PC4,axis=1),np.mean(MR_PC5,axis=1),np.mean(MR_PC6,axis=1),np.mean(MR_PC7,axis=1),np.mean(MR_PC8,axis=1),np.mean(MR_PC9,axis=1),np.mean(MR_PC10,axis=1),np.mean(MR_PC11,axis=1),np.mean(MR_PC12,axis=1),np.mean(MR_PC13,axis=1),np.mean(MR_PC14,axis=1)))

# imLR = np.vstack((np.nanmean(LR_PV1,axis=1),np.nanmean(LR_PV2,axis=1),np.nanmean(LR_PV3,axis=1),np.nanmean(LR_PV4,axis=1),np.nanmean(LR_PV5,axis=1),np.nanmean(LR_PV6,axis=1),np.nanmean(LR_PV7,axis=1),np.nanmean(LR_PV8,axis=1),np.nanmean(LR_PV9,axis=1),np.nanmean(LR_PV10,axis=1),np.nanmean(LR_PV11,axis=1),np.nanmean(LR_PV12,axis=1),np.nanmean(LR_PV13,axis=1),np.nanmean(LR_PV14,axis=1)))

# imMR = np.vstack((np.nanmean(MR_PC1,axis=1),np.nanmean(MR_PC2,axis=1),np.nanmean(MR_PC3,axis=1),np.nanmean(MR_PC4,axis=1),np.nanmean(MR_PC5,axis=1),np.nanmean(MR_PC6,axis=1),np.nanmean(MR_PC7,axis=1),np.nanmean(MR_PC8,axis=1),np.nanmean(MR_PC9,axis=1),np.nanmean(MR_PC10,axis=1),np.nanmean(MR_PC11,axis=1),np.nanmean(MR_PC12,axis=1),np.nanmean(MR_PC13,axis=1),np.nanmean(MR_PC14,axis=1)))

# imLR = np.vstack((np.nanmean(LR_PC1,axis=1),np.nanmean(LR_PC2,axis=1),np.nanmean(LR_PC3,axis=1),np.nanmean(LR_PC4,axis=1),np.nanmean(LR_PC5,axis=1),np.nanmean(LR_PC6,axis=1),np.nanmean(LR_PC7,axis=1),np.nanmean(LR_PC8,axis=1),np.nanmean(LR_PC9,axis=1),np.nanmean(LR_PC10,axis=1),np.nanmean(LR_PC11,axis=1),np.nanmean(LR_PC12,axis=1),np.nanmean(LR_PC13,axis=1),np.nanmean(LR_PC14,axis=1)))


# hght = np.arange(0.5,14.5,1)

# sed_load = pd.read_csv('../raw_data/time_series/Elwha_DailySedimentLoads_2011to2016.csv')

# sed_load = sed_load[['Day',
# 'Daily Discharge (m3/s)',
# 'Total sediment discharge (tonnes)',
# 'Ave fraction fines (based on two turbidimeters)']]

# dt_sed = [datetime.strptime(time,'%m/%d/%Y') for time in sed_load['Day']]
# dt_sed = np.array(dt_sed)

# ind = np.argsort(dt_sed)

# t_sed = np.array([float(d.strftime('%s')) for d in dt_sed[ind]])
# t =  np.array([float(d.strftime('%s')) for d in dt])

# # O_MR = np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values)
# OS = np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values)

# OQ = np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values)


# ########################################
# plt.figure(figsize=(12,12))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(221)
# plt.plot(dt[1:],np.nanmean(MR_PC,axis=1),'k-',label='MR')
# plt.plot(dt[1:],np.nanmean(LR_PC,axis=1),'r--',label='LR')
# plt.legend()
# plt.ylabel(r"Mean autocorrelation coefficient")
# plt.title('a) ', loc='left'); 

# plt.subplot(222)
# x=np.nanmax(imMR,axis=1)
# x[np.isnan(x)]=0.0
# plt.plot(x,hght, color='k', linestyle='-',label='MR')

# x=np.nanmax(imLR,axis=1)
# x[np.isnan(x)]=0.0
# plt.plot(x,hght, color='r', linestyle='--',label='LR')
# plt.ylim(hght[0],hght[-1])
# plt.ylabel("Height (m)"); plt.xlabel(r"Mean autocorrelation coefficient")
# plt.title('b) ', loc='left'); 

# plt.subplot(223)
# df_mnth_Q = sed_load.groupby(pd.PeriodIndex(sed_load['Day'], freq="M"))['Daily Discharge (m3/s)'].mean()
# plt.plot(dt,OQ,'ro',label='Aerial imagery')
# df_mnth_Q.plot(label='Monthly mean\n discharge')
# plt.ylabel(r"Water discharge (m/s$^3$)"); #plt.xlabel(r"Mean Autocorrelation coefficient")
# plt.xlabel('')
# plt.legend()
# plt.title('c) ', loc='left');

# plt.subplot(224)
# df_mnth_L = sed_load.groupby(pd.PeriodIndex(sed_load['Day'], freq="M"))['Total sediment discharge (tonnes)'].mean()
# plt.plot(dt,OS,'ro',label='Aerial imagery')
# df_mnth_L.plot(label='Monthly mean\n sediment load')
# plt.ylabel("Sediment load (tonnes)"); #plt.xlabel(r"Mean Autocorrelation coefficient")
# plt.xlabel('')
# plt.legend(loc=2)
# plt.title('d)', loc='left');

# # plt.show()
# plt.savefig("summaries/LR_MR_wood_persistence_perHeight_bin.png", dpi=300, bbox_inches="tight")
# plt.close()


######################################################
######################################################





# ########################################
# plt.figure(figsize=(16,16))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(421)
# plt.imshow(np.flipud(LR_IC1), cmap='inferno', extent=[0, LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 0m<= h <0.5m")
# plt.gca().invert_yaxis()
# plt.title('a) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(422)
# plt.imshow(np.flipud(LR_IC2), cmap='inferno', extent=[0, LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 0.5m<= h <1.5m")
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(423)
# plt.imshow(np.flipud(LR_IC3), cmap='inferno', extent=[0, LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 1.5m<= h <2.5m")
# plt.gca().invert_yaxis()
# plt.title('c) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(424)
# plt.imshow(np.flipud(LR_IC4), cmap='inferno', extent=[0, LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 2.5m<= h <3.5m")
# plt.gca().invert_yaxis()
# plt.title('d) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(425)
# plt.imshow(np.flipud(LR_IC5), cmap='inferno', extent=[0, LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 3.5m<= h <4.5m")
# plt.gca().invert_yaxis()
# plt.title('e) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(426)
# plt.imshow(np.flipud(LR_IC6), cmap='inferno', extent=[0, LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 4.5m<= h <5.5m")
# plt.gca().invert_yaxis()
# plt.title('f) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(427)
# plt.imshow(np.flipud(LR_IC7), cmap='inferno', extent=[0, LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 5.5m<= h <6.5m")
# plt.gca().invert_yaxis()
# plt.title('g) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(428)
# plt.imshow(np.flipud(LR_IC8), cmap='inferno', extent=[0, LR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 6.5m<= h <8m")
# plt.gca().invert_yaxis()
# plt.title('h) ', loc='left'); plt.xlabel("River kilometer(km)"); 


# # plt.show()
# plt.savefig("summaries/LR_wood_spacetime_intersection_perHeight_bin.png", dpi=300, bbox_inches="tight")
# plt.close()



# ########################################
# plt.figure(figsize=(16,16))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(421)
# plt.imshow(np.flipud(MR_IC1), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 0m<= h <0.5m")
# plt.gca().invert_yaxis()
# plt.title('a) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(422)
# plt.imshow(np.flipud(MR_IC2), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 0.5m<= h <1.5m")
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(423)
# plt.imshow(np.flipud(MR_IC3), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 1.5m<= h <2.5m")
# plt.gca().invert_yaxis()
# plt.title('c) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(424)
# plt.imshow(np.flipud(MR_IC4), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 2.5m<= h <3.5m")
# plt.gca().invert_yaxis()
# plt.title('d) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(425)
# plt.imshow(np.flipud(MR_IC5), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 3.5m<= h <4.5m")
# plt.gca().invert_yaxis()
# plt.title('e) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(426)
# plt.imshow(np.flipud(MR_IC6), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 4.5m<= h <5.5m")
# plt.gca().invert_yaxis()
# plt.title('f) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(427)
# plt.imshow(np.flipud(MR_IC7), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 5.5m<= h <6.5m")
# plt.gca().invert_yaxis()
# plt.title('g) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(428)
# plt.imshow(np.flipud(MR_IC8), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Intersection coefficient, 6.5m<= h <8m")
# plt.gca().invert_yaxis()
# plt.title('h) ', loc='left'); plt.xlabel("River kilometer(km)"); 


# # plt.show()
# plt.savefig("summaries/MR_wood_spacetime_intersection_perHeight_bin.png", dpi=300, bbox_inches="tight")
# plt.close()

# ########################################
# plt.figure(figsize=(16,16))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(421)
# plt.imshow(np.flipud(MR_MC1), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Co-location coefficient, 0m<= h <0.5m")
# plt.gca().invert_yaxis()
# plt.title('a) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(422)
# plt.imshow(np.flipud(MR_MC2), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Co-location coefficient, 0.5m<= h <1.5m")
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(423)
# plt.imshow(np.flipud(MR_MC3), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Co-location coefficient, 1.5m<= h <2.5m")
# plt.gca().invert_yaxis()
# plt.title('c) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(424)
# plt.imshow(np.flipud(MR_MC4), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Co-location coefficient, 2.5m<= h <3.5m")
# plt.gca().invert_yaxis()
# plt.title('d) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(425)
# plt.imshow(np.flipud(MR_MC5), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Co-location coefficient, 3.5m<= h <4.5m")
# plt.gca().invert_yaxis()
# plt.title('e) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(426)
# plt.imshow(np.flipud(MR_MC6), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Co-location coefficient, 4.5m<= h <5.5m")
# plt.gca().invert_yaxis()
# plt.title('f) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(427)
# plt.imshow(np.flipud(MR_MC7), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Co-location coefficient, 5.5m<= h <6.5m")
# plt.gca().invert_yaxis()
# plt.title('g) ', loc='left'); plt.xlabel("River kilometer(km)"); 

# plt.subplot(428)
# plt.imshow(np.flipud(MR_MC8), cmap='inferno', extent=[0, MR[-1] , dt[1], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Co-location coefficient, 6.5m<= h <8m")
# plt.gca().invert_yaxis()
# plt.title('h) ', loc='left'); plt.xlabel("River kilometer(km)"); 


# # plt.show()
# plt.savefig("summaries/MR_wood_spacetime_colocation_perHeight_bin.png", dpi=300, bbox_inches="tight")
# plt.close()



        # f, ax = plt.subplots(
        #     1, 2, figsize=(9, 3), gridspec_kw=dict(width_ratios=(6, 3))
        # )

        # # plot all the simulations with very fine lines
        # ax[0].plot(
        #     f_test.support, f_test.simulations.T, color="k", alpha=0.01
        # )
        # # and show the average of simulations
        # ax[0].plot(
        #     f_test.support,
        #     np.median(f_test.simulations, axis=0),
        #     color="cyan",
        #     label="median simulation",
        # )


        # # and the observed pattern's F function
        # ax[0].plot(
        #     f_test.support, f_test.statistic, label="observed", color="red"
        # )

        # # clean up labels and axes
        # ax[0].set_xlabel("distance")
        # ax[0].set_ylabel("% of nearest point in pattern\ndistances shorter")
        # ax[0].legend()
        # ax[0].set_xlim(0, 5)
        # ax[0].set_title(r"Ripley's $F(d)$ function")

        # # plot the pattern itself on the next frame
        # ax[1].scatter(*dat.T)

        # # and clean up labels and axes there, too
        # ax[1].set_xticks([])
        # ax[1].set_yticks([])
        # ax[1].set_xticklabels([])
        # ax[1].set_yticklabels([])
        # ax[1].set_title("Pattern")
        # f.tight_layout()
        # plt.show()



        ## if g_index_obs>g_index_random, then the observed pattern is closer to their nearest neighbors than 
        # would be expected from a completely spatially random pattern. The pattern is clustered.


        # f, ax = plt.subplots(
        #     1, 2, figsize=(9, 3), gridspec_kw=dict(width_ratios=(6, 3))
        # )
        # # plot all the simulations with very fine lines
        # ax[0].plot(
        #     g_test.support, g_test.simulations.T, color="k", alpha=0.01
        # )
        # # and show the average of simulations
        # ax[0].plot(
        #     g_test.support,
        #     np.median(g_test.simulations, axis=0),
        #     color="cyan",
        #     label="median simulation",
        # )

        # # and the observed pattern's G function
        # ax[0].plot(
        #     g_test.support, g_test.statistic, label="observed", color="red"
        # )

        # # clean up labels and axes
        # ax[0].set_xlabel("distance")
        # ax[0].set_ylabel("% of nearest neighbor\ndistances shorter")
        # ax[0].legend()
        # ax[0].set_xlim(0, 5)
        # ax[0].set_title(r"Ripley's $G(d)$ function")

        # # plot the pattern itself on the next frame
        # ax[1].scatter(*dat.T)

        # # and clean up labels and axes there, too
        # ax[1].set_xticks([])
        # ax[1].set_yticks([])
        # ax[1].set_xticklabels([])
        # ax[1].set_yticklabels([])
        # ax[1].set_title("Pattern")
        # f.tight_layout()
        # plt.show()


        # k_test = distance_statistics.k_test(dat[::5], keep_simulations=True)

        # plt.plot(k_test.support, k_test.simulations.T, color='k', alpha=.01)
        # plt.plot(k_test.support, k_test.statistic, color='orangered')

        # plt.scatter(k_test.support, k_test.statistic, 
        #             cmap='viridis', c=k_test.pvalue < .05,
        #             zorder=4 # make sure they plot on top
        #         )

        # plt.xlabel('Distance')
        # plt.ylabel('K Function')
        # plt.title('K Function Plot')
        # plt.show()


        # l_test = distance_statistics.l_test(dat[::5], keep_simulations=True)

        # plt.plot(l_test.support, l_test.simulations.T, color='k', alpha=.01)
        # plt.plot(l_test.support, l_test.statistic, color='orangered')

        # plt.scatter(l_test.support, l_test.statistic, 
        #             cmap='viridis', c=l_test.pvalue < .05,
        #             zorder=4 # make sure they plot on top
        #         )

        # plt.xlabel('Distance')
        # plt.ylabel('L Function')
        # plt.title('L Function Plot')
        # plt.show()



# import pandas as pd 
# import itertools
# # list(itertools.chain.from_iterable(a))

    
# dat = np.vstack(dat)

# dbLR = pd.DataFrame.from_records(LR_props)
# coordinates = dbLR[["x", "y"]]
