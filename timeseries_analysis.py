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
from area import area
from skimage.measure import label, regionprops_table
from functools import partial

from skimage.restoration import inpaint
from skimage.restoration import denoise_wavelet
from scipy.optimize import curve_fit

from matplotlib.patches import Rectangle

# rescale_sigma=True required to silence deprecation warnings
_denoise_wavelet = partial(denoise_wavelet, rescale_sigma=True)


def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    Code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn


def props_df(image):
    label_img = label(image)

    props = regionprops_table(label_img, properties=('area',
                                                    'centroid',
                                                    'equivalent_diameter_area',
                                                    'solidity',
                                                    'axis_major_length',
                                                    'axis_minor_length'))

    return pd.DataFrame(props)


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


#################### size dfistributions

### LR
F_LR=[] 
for time in tqdm(times):
    tmp = LRwood_geotiffs_ds.wood.sel(time=time)

    props = props_df(tmp.to_numpy())

    frq, bins, ax = plt.hist(props.area.values, bins=np.linspace(1,40000,100))
    del ax
    F_LR.append(frq)

F_LR = np.array(F_LR)


### MR
F_MR=[] 
for time in tqdm(times):
    tmp = MRwood_geotiffs_ds.wood.sel(time=time)

    props = props_df(tmp.to_numpy())

    frq, bins, ax = plt.hist(props.area.values, bins=np.linspace(1,40000,100))
    del ax
    F_MR.append(frq)

F_MR = np.array(F_MR)


#####################################################################

dists = pd.read_csv('br_dists.csv')
LR = np.hstack((0,np.array(dists['LR'])))
MR = np.hstack((0,np.array(dists['MR'][:43])))

## rescale distances
LR = rescale_array(LR,11,2)
MR = rescale_array(MR[::-1],12,20)

########################################

grid2sqm = 64
bins=np.linspace(1,40000,100)/grid2sqm

mask = np.isfinite(np.log(F_MR))
F_MRi = inpaint.inpaint_biharmonic(np.log(F_MR), ~mask)

mask = np.isfinite(np.log(F_LR))
F_LRi = inpaint.inpaint_biharmonic(np.log(F_LR), ~mask)


########################################
plt.figure(figsize=(10,13))
plt.subplots_adjust(wspace=0.2, hspace=0.2)

plt.subplot(321)
plt.imshow(np.flipud(F_MRi), cmap='inferno', extent=[bins[0], bins[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label("Log frequency")
plt.xscale('log')
plt.title('a)', loc='left')
plt.xlabel(r"Wood pile or piece area (m$^2$)")

y = F_MRi.copy()/np.nansum(F_MRi,axis=0)/len(times)
x = -.5
r_v = (y*bins[1:]**x) / np.nansum(y*bins[1:]**x) #volume-by-weight proportion
mnsz = np.nansum(r_v * bins[1:],axis=1)
plt.semilogx(mnsz, dt, 'w-o',lw=2, label=r'Mean size')

sig=[]
for counter in range(len(mnsz)):
    sig.append(np.sqrt(np.nansum(y[counter,:]*((bins[1:]-mnsz[counter])**2))))

sig = np.array(sig)/mnsz
# plt.plot(sig, dt, 'w-',lw=4)
plt.plot(sig, dt, 'w--',lw=2, label=r'Coefficient of variation')
plt.xlim(5,625)
plt.gca().invert_yaxis()
plt.legend()
# plt.gca().annotate("", xy=(2012, 12), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))

plt.subplot(322)
plt.imshow(np.flipud(F_LRi), cmap='inferno', extent=[bins[0], bins[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label("Log frequency")
plt.xscale('log')
plt.title('b)', loc='left')
plt.xlabel(r"Wood pile or piece area (m$^2$)")

y = F_LRi.copy()/np.nansum(F_LRi,axis=0)/len(times)
x = -.5
r_v = (y*bins[1:]**x) / np.nansum(y*bins[1:]**x) #volume-by-weight proportion
mnsz = np.nansum(r_v * bins[1:],axis=1)
plt.semilogx(mnsz, dt, 'w-o',lw=2)

sig=[]
for counter in range(len(mnsz)):
    sig.append(np.sqrt(np.nansum(y[counter,:]*((bins[1:]-mnsz[counter])**2))))

sig = np.array(sig)/mnsz
plt.plot(sig, dt, 'w--',lw=2, label=r'Coefficient of variation')
plt.xlim(5,625)
plt.gca().invert_yaxis()

plt.subplot(312)
plt.plot(bins[1:], np.mean(np.flipud(F_MRi),axis=0), alpha=1, color='k', label='MR')
plt.plot(bins[1:], np.mean(np.flipud(F_LRi),axis=0),'r--',alpha=1,  label='LR')
popt, pcov = curve_fit(func, bins[1:], np.mean(np.flipud(F_MRi),axis=0))
plt.plot(bins[1:], func(bins[1:], *popt), 'b-', label='y='+str(popt[0])[:3]+r'e^{'+str(popt[1])[:5]+'x}'+'+'+str(popt[2])[:4])
# plt.xscale('log')
plt.xlabel(r"Mean wood pile or piece area (m$^2$)")
plt.title('c)', loc='left')
plt.legend()

plt.subplot(313)
y = F_MRi.copy()/np.nansum(F_MRi,axis=0)/len(times)
x = -.5
r_v = (y*bins[1:]**x) / np.nansum(y*bins[1:]**x) #volume-by-weight proportion
mnsz1 = np.nansum(r_v * bins[1:],axis=1)
plt.plot(mnsz1, dt, 'ko',lw=1, label='MR')

y = F_LRi.copy()/np.nansum(F_LRi,axis=0)/len(times)
x = -.5
r_v = (y*bins[1:]**x) / np.nansum(y*bins[1:]**x) #volume-by-weight proportion
mnsz2 = np.nansum(r_v * bins[1:],axis=1)
plt.plot(mnsz2, dt, 'rs',lw=1, label='LR')

plt.plot((mnsz1+mnsz2)/2, dt, 'b',lw=2, label='Inter-reach mean')

plt.gca().invert_yaxis()
plt.xlim(0,35)
plt.xlabel(r"Wood pile or piece area (m$^2$)")
plt.title('d)', loc='left')
plt.legend(loc=0)
# plt.show()

plt.savefig("summaries/MR_LR_wood_size_history.png", dpi=300, bbox_inches="tight")
plt.close()


with np.load('summaries/Wood_time_series_largepieces_smallpieces.npz', allow_pickle=True) as f:
    LR_BRarr_large = f['LR_BRarr_large']
    MR_BRarr_large = f['MR_BRarr_large']
    dt = f['dt']
    grid2sqm = f['grid2sqm']
    LR_BRarr_small = f['LR_BRarr_small']
    MR_BRarr_small = f['MR_BRarr_small']


with np.load('summaries/Wood_time_series.npz', allow_pickle=True) as f:
    LR_BRarr = f['LR_BRarr']
    MR_BRarr = f['MR_BRarr']
    dt = f['dt']
    grid2sqm = f['grid2sqm']


with np.load('summaries/Sed_time_series.npz', allow_pickle=True) as f:
    LR_BRarrsed = f['LR_BRarrsed']
    MR_BRarrsed = f['MR_BRarrsed']


########################################
plt.figure(figsize=(16,20))
plt.subplots_adjust(wspace=0.2, hspace=0.2)

plt.subplot(421)
plt.plot(MR, np.sum(MR_BRarr,axis=0),'k-', label='MR', lw=2)

sigma = (np.sum(MR_BRarr,axis=0)/100)*15
X1_plus_sigma = np.sum(MR_BRarr,axis=0) + sigma
X1_minus_sigma = np.sum(MR_BRarr,axis=0) - sigma

plt.fill_between(MR, X1_plus_sigma, X1_minus_sigma, alpha = 0.2, color = [.5,.5,.5])

plt.plot(LR, np.sum(LR_BRarr,axis=0),'r--', label='LR', lw=2)

sigma = (np.sum(LR_BRarr,axis=0)/100)*15
X1_plus_sigma = np.sum(LR_BRarr,axis=0) + sigma
X1_minus_sigma = np.sum(LR_BRarr,axis=0) - sigma

plt.fill_between(LR, X1_plus_sigma, X1_minus_sigma, alpha = 0.2, color = [.5,.5,.5])

plt.ylim(0,100+np.maximum(np.sum(MR_BRarr,axis=0).max(), np.sum(LR_BRarr,axis=0).max()))

rec=Rectangle((11,0), 1, np.maximum(np.sum(MR_BRarr,axis=0).max(), np.sum(LR_BRarr,axis=0).max()), clip_on=False, color='gray')
plt.gca().add_artist(rec)

plt.xlabel("Distance downstream (km)"); 
plt.ylabel(r"Sum of estimated" "\n" r"wood m$^2$")
plt.gca().invert_xaxis()
plt.legend()
plt.title('a) ', loc='left')
plt.text(11,25000,'former\nLake\nAldwell')

plt.subplot(422)
plt.plot(MR, np.sum(MR_BRarr+MR_BRarrsed,axis=0),'k-', label='MR')
plt.plot(LR, np.sum(LR_BRarr+LR_BRarrsed,axis=0),'r--', label='LR')
plt.ylabel(r"Sum of estimated" "\n" r"sediment m$^2$")
plt.xlabel("Distance downstream (km)"); 

plt.ylim(0,np.maximum(np.sum(LR_BRarr+LR_BRarrsed,axis=0).max(), np.sum(LR_BRarr+LR_BRarrsed,axis=0).max()))

rec=Rectangle((11,0), 1, np.maximum(np.sum(LR_BRarr+LR_BRarrsed,axis=0).max(), np.sum(LR_BRarr+LR_BRarrsed,axis=0).max()), clip_on=False, color='gray')
plt.gca().add_artist(rec)

plt.legend()
plt.title('b) ', loc='left')
plt.gca().invert_xaxis()
plt.legend()

# plt.show()
plt.savefig("summaries/sedimentwood_space_plots.png", dpi=300, bbox_inches="tight")
plt.close()





####################################################################


plt.figure(figsize=(12,12))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.subplot(221)
plt.plot(dt, np.sum(MR_BRarr,axis=1),'k-', label='MR', lw=2)

sigma = (np.sum(MR_BRarr,axis=1)/100)*15
X1_plus_sigma = np.sum(MR_BRarr,axis=1) + sigma
X1_minus_sigma = np.sum(MR_BRarr,axis=1) - sigma

plt.fill_between(dt, X1_plus_sigma, X1_minus_sigma, alpha = 0.2, color = 'k')
plt.plot(dt, np.sum(LR_BRarr,axis=1),'r--', label='LR', lw=2)

sigma = (np.sum(LR_BRarr,axis=1)/100)*15
X1_plus_sigma = np.sum(LR_BRarr,axis=1) + sigma
X1_minus_sigma = np.sum(LR_BRarr,axis=1) - sigma

plt.fill_between(dt, X1_plus_sigma, X1_minus_sigma, alpha = 0.2, color = 'r')
plt.ylim(0,5000+np.maximum(np.sum(MR_BRarr,axis=1).max(), np.sum(LR_BRarr,axis=1).max()))
plt.ylabel(r"Sum of estimated" "\n" r"wood (m$^2$)")
# plt.gca().invert_xaxis()
plt.legend()
plt.title('a) ', loc='left')

plt.subplot(222)
plt.plot(dt, np.sum(MR_BRarr+MR_BRarrsed,axis=1),'k-', label='MR')
plt.plot(dt, np.sum(LR_BRarr+LR_BRarrsed,axis=1),'r--', label='LR')
plt.ylabel(r"Sum of estimated" "\n" r"sediment (m$^2$)")

plt.ylim(0,400000)
plt.legend()
plt.title('b) ', loc='left')
# plt.gca().invert_xaxis()
plt.legend()

woodsed_ratio_MR = np.sum(MR_BRarr,axis=1)/(np.sum(MR_BRarrsed,axis=1)+np.sum(MR_BRarr,axis=1))
woodsed_ratio_LR = np.sum(LR_BRarr,axis=1)/(np.sum(LR_BRarrsed,axis=1)+np.sum(LR_BRarr,axis=1))

plt.subplot(223)
plt.plot(dt,woodsed_ratio_MR,'k-', label='MR')
plt.plot(dt,woodsed_ratio_LR,'r--', label='LR')
plt.ylabel("Ratio of \n wood and sediment (-)");
plt.legend()
plt.title(r'c)', loc='left') 
plt.ylim(0,0.2) #.15+woodsed_ratio_LR.max())

woodsed_ratio_MR = (np.sum(MR_BRarr,axis=1)/np.sum(MR_BRarr,axis=1).max() ) / ((np.sum(MR_BRarrsed,axis=1)+np.sum(MR_BRarr,axis=1)) / (np.sum(MR_BRarrsed,axis=1).max()+np.sum(MR_BRarr,axis=1).max()))
woodsed_ratio_LR = (np.sum(LR_BRarr,axis=1)/np.sum(LR_BRarr,axis=1).max() ) / ((np.sum(LR_BRarrsed,axis=1)+np.sum(LR_BRarr,axis=1)) / (np.sum(LR_BRarrsed,axis=1).max()+np.sum(LR_BRarr,axis=1).max()))

plt.subplot(224)
plt.plot(dt,woodsed_ratio_MR,'k-', label='MR')
plt.plot(dt,woodsed_ratio_LR,'r--', label='LR')
plt.ylabel("Ratio of normalized\n wood and sediment (-)");
plt.legend()
plt.title(r'd)', loc='left')
plt.ylim(0,1.7) #.15+woodsed_ratio_LR.max())

# plt.show()
plt.savefig("summaries/sedimentwoodratio_time_plots.png", dpi=300, bbox_inches="tight")
plt.close()




#### divide out by area of each BR for a wood concentration\
A_MR = np.array(A_MR)
A_LR = np.array(A_LR)

## wood
MR_BRarr_c = MR_BRarr/A_MR
LR_BRarr_c = LR_BRarr/A_LR

## sed
MR_BRarrsed_c = (MR_BRarrsed+MR_BRarr)/A_MR
LR_BRarrsed_c = (LR_BRarrsed+LR_BRarr)/A_LR

wvmax = np.maximum(np.max(MR_BRarr_c.flatten()), np.max(LR_BRarr_c.flatten()))
svmax = np.maximum(np.max(MR_BRarrsed_c.flatten()), np.max(LR_BRarrsed_c.flatten()))
smax = np.maximum(np.max((MR_BRarrsed+MR_BRarr).flatten()), np.max((LR_BRarrsed+LR_BRarr).flatten()))
wmax = np.maximum(np.max((MR_BRarr).flatten()), np.max((LR_BRarr).flatten()))


########################################
plt.figure(figsize=(14,20))
plt.subplots_adjust(wspace=0.2, hspace=0.2)

plt.subplot(421)
plt.imshow(np.flipud(MR_BRarr), cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=wmax)
cb=plt.colorbar(); cb.set_label(r"Wood area, m$^2$")
plt.gca().invert_yaxis()
plt.title('a)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

plt.subplot(422)
plt.imshow(np.flipud(LR_BRarr), cmap='inferno', extent=[LR[0] , LR[-1], dt[0], dt[-1]], aspect='auto', vmin=0, vmax=wmax)
cb=plt.colorbar(); cb.set_label(r"Wood area, m$^2$")
plt.gca().invert_yaxis()
# plt.title('c) LR', loc='left')
# plt.xlabel("Distance downstream (km)"); 

plt.subplot(423)
plt.imshow(np.flipud(MR_BRarr+MR_BRarrsed), cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=smax)
cb=plt.colorbar(); cb.set_label(r"Sediment area, m$^2$")
plt.gca().invert_yaxis()
plt.title('b)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

plt.subplot(424)
plt.imshow(np.flipud(LR_BRarr+LR_BRarrsed), cmap='inferno', extent=[LR[0] , LR[-1], dt[0], dt[-1]], aspect='auto', vmin=0, vmax=smax)
cb=plt.colorbar(); cb.set_label(r"Sediment area, m$^2$")
plt.gca().invert_yaxis()
# plt.title('d) LR', loc='left')
# plt.xlabel("Distance downstream (km)"); 

plt.subplot(425)
plt.imshow(np.flipud(MR_BRarr_c), cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=wvmax)
cb=plt.colorbar(); cb.set_label("Normalized wood area\n" r"m$^2$/m$^2$")
plt.gca().invert_yaxis()
plt.title('c)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

plt.subplot(427)
plt.imshow(np.flipud(MR_BRarrsed_c), cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=svmax)
cb=plt.colorbar(); cb.set_label("Normalized sediment area\n" r"m$^2$/m$^2$")
plt.gca().invert_yaxis()
# plt.title('g) MR', loc='left')
plt.xlabel("River kilometer"); 

plt.subplot(426)
plt.imshow(np.flipud(LR_BRarr_c), cmap='inferno', extent=[LR[0], LR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=wvmax)
cb=plt.colorbar(); cb.set_label("Normalized wood area\n" r"m$^2$/m$^2$")
plt.gca().invert_yaxis()
plt.title('d)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

plt.subplot(428)
plt.imshow(np.flipud(LR_BRarrsed_c), cmap='inferno', extent=[LR[0], LR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=svmax)
cb=plt.colorbar(); cb.set_label("Normalized sediment area\n" r"m$^2$/m$^2$")
plt.gca().invert_yaxis()
# plt.title('d)', loc='left');
plt.xlabel("River kilometer"); 

# plt.show()
plt.savefig("summaries/sedimentwood_spacetime_mag_conc.png", dpi=300, bbox_inches="tight")
plt.close()




########################################
plt.figure(figsize=(14,20))
plt.subplots_adjust(wspace=0.2, hspace=0.2)

X = []
for k in range(MR_BRarr_c.shape[0]):
    X.append(np.correlate(MR_BRarr_c[k,:], MR_BRarr_c[k,:]+MR_BRarrsed_c[k,:], 'same'))

X = np.vstack(X)

plt.subplot(421)
plt.imshow(np.flipud(X), cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=1)
cb=plt.colorbar(); cb.set_label("Cross-correlation of\nwood and sediment area")
plt.gca().invert_yaxis()
plt.title('a)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

X = []
for k in range(LR_BRarr_c.shape[0]):
    X.append(np.correlate(LR_BRarr_c[k,:], LR_BRarr_c[k,:]+LR_BRarrsed_c[k,:], 'same'))

X = np.vstack(X)

plt.subplot(422)
plt.imshow(np.flipud(X), cmap='inferno', extent=[LR[0] , LR[-1], dt[0], dt[-1]], aspect='auto', vmin=0, vmax=1)
cb=plt.colorbar(); cb.set_label("Cross-correlation of\nwood and sediment area")
plt.gca().invert_yaxis()
# plt.title('c) LR', loc='left')
# plt.xlabel("Distance downstream (km)"); 


plt.show()




##############################################################

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



###################################################

fig, ax = plt.subplots(nrows=2, ncols=3)
fig.set_size_inches(16,12)
plt.subplots_adjust(wspace=0.6, hspace=0.2)

####### flow
# plt.subplot(221)
ax[0][0].plot(dt, np.sum(MR_BRarr,axis=1),'k-', label='MR wood')
ax[0][0].plot(dt, np.sum(LR_BRarr,axis=1),'r--', label='LR wood')
ax[0][0].set_ylabel('Total wood area (m2)')

ax2 = ax[0][0].twinx()
ax2.plot(dt_sed[ind], sed_load['Daily Discharge (m3/s)'][ind],'b-', alpha=0.5, label='Daily discharge')
ax2.set_ylabel(r'Daily Discharge (m$^3$/s)', color='b')
ax[0][0].set_title('a) ', loc='left')

ax2.plot(dt, OQ, 'b-o') #, legend='Discharge at image acquisition'

###############
ax[0][1].plot(dt_sed[ind], sed_load['Total sediment discharge (tonnes)'][ind],'-', color=[.5,.5,.5], alpha=0.5)
ax[0][1].set_ylabel(r'Total sediment discharge (tonnes)', color=[.5,.5,.5])
ax[0][1].set_title('b) ', loc='left')
ax[0][1].plot(dt, OS, '-o', color=[.5,.5,.5]) #, legend='Discharge at image acquisition'

ax3 = ax[0][1].twinx()
ax3.plot(dt_sed[ind], sed_load['Ave fraction fines (based on two turbidimeters)'][ind],'b-', alpha=0.5)
ax3.set_ylabel(r'Average fraction of fines', color='b')


####### flow
f,bins=np.histogram(sed_load['Daily Discharge (m3/s)'][ind].values)
ax[0][2].plot(bins[:-1],f/f.max(),'k')
data = np.hstack([np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values)]*2)
data = np.hstack((data, np.random.randint(data.min(), data.max(),len(data))))
f,bins=np.histogram(data)
ax[0][2].bar(bins[:-1],f/f.max(),width=2,color='r')
# ax[0][2].set_ylabel('Total wood area (m2)')

ax[0][2].set_ylabel(r"Normalized frequency")
ax[0][2].set_xlabel(r'Daily Discharge (m$^3$/s)', color='k')
ax[0][2].set_title('c) ', loc='left')

###############
ax[1][0].plot(np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values), np.sum(MR_BRarr,axis=1), 'bo')
ax[1][0].plot(np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values), np.sum(LR_BRarr,axis=1), 'rs')

A = np.vstack([OQ, np.ones(len(OQ))]).T
E = np.sum(LR_BRarr,axis=1)
m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
ax[1][0].plot(OQ, m*OQ+ c, 'r:',lw=2, label='(LR) y = '+str(m)[:4]+'x+'+str(c)[:4])
ax[1][0].text(20,45000,r'R$^2$ = '+str(np.min(np.corrcoef(OQ,E))**2)[:6], color='r')

E = np.sum(MR_BRarr,axis=1)
m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
ax[1][0].plot(OQ, m*OQ+ c, 'b:',lw=2, label='(MR) y = '+str(m)[:4]+'x+'+str(c)[:4])
ax[1][0].text(20,40000,r'R$^2$ = '+str(np.min(np.corrcoef(OQ,E))**2)[:6], color='b')

ax[1][0].legend(fontsize=7)
ax[1][0].set_ylabel(r"Estimated wood, m$^2$")
ax[1][0].set_xlabel(r'Discharge, day of aerial survey  (m$^3$/s)')
ax[1][0].set_title('d) ', loc='left')


###############
ax[1][1].scatter(np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values), np.sum(MR_BRarr,axis=1), 50, np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values), cmap='inferno', edgecolor='k', lw=1)

imc = ax[1][1].scatter(np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values), np.sum(LR_BRarr,axis=1), 50, np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values), cmap='inferno', edgecolor='k', lw=1)

cbar = plt.colorbar(imc)
cbar.set_label('Total sediment discharge,\n day of aerial survey (tonnes)')

ax[1][1].set_ylabel(r"Estimated wood, m$^2$")
ax[1][1].set_xlabel(r'Discharge, day of aerial survey  (m$^3$/s)')
ax[1][1].set_title('e) ', loc='left')

woodsed_ratio_MR = (np.sum(MR_BRarr,axis=1)/np.sum(MR_BRarr,axis=1).max() ) / ((np.sum(MR_BRarrsed,axis=1)+np.sum(MR_BRarr,axis=1)) / (np.sum(MR_BRarrsed,axis=1).max()+np.sum(MR_BRarr,axis=1).max()))
woodsed_ratio_LR = (np.sum(LR_BRarr,axis=1)/np.sum(LR_BRarr,axis=1).max() ) / ((np.sum(LR_BRarrsed,axis=1)+np.sum(LR_BRarr,axis=1)) / (np.sum(LR_BRarrsed,axis=1).max()+np.sum(LR_BRarr,axis=1).max()))

###############
# # ax[1][1].plot(np.log(O_MR),woodsed_ratio_MR,'ko',label="MR")
ax[1][2].plot(np.log(OS)[summer],woodsed_ratio_MR[summer],'bo',label='MR, Discharge < 30 m$^3$/s')
ax[1][2].plot(np.log(OS)[winter],woodsed_ratio_MR[winter],'bo',label='MR, Discharge > 30 m$^3$/s')

A = np.vstack([np.log(OS), np.ones(len(OS))]).T
m, c = np.linalg.lstsq(A, woodsed_ratio_MR, rcond=None)[0]
ax[1][2].plot(np.log(OS), m*np.log(OS)+ c, 'b-',lw=2, label='(MR) y = '+str(m)[:4]+'x+'+str(c)[:4])
ax[1][2].text(1,1.6,r'R$^2$ (MR) = '+str(np.min(np.corrcoef(np.log(OS),woodsed_ratio_MR))**2)[:6], color='b')


###############
# ax[1][1].plot(np.log(O_LR),woodsed_ratio_LR,'ko',label="LR")
ax[1][2].plot(np.log(OS)[summer],woodsed_ratio_LR[summer],'rs',label='LR, Discharge < 30 m$^3$/s')
ax[1][2].plot(np.log(OS)[winter],woodsed_ratio_LR[winter],'rs',label='LR, Discharge > 30 m$^3$/s')

A = np.vstack([np.log(OS), np.ones(len(OS))]).T
m, c = np.linalg.lstsq(A, woodsed_ratio_LR, rcond=None)[0]
ax[1][2].plot(np.log(OS), m*np.log(OS)+ c, 'r-',lw=2, label='(LR) y = '+str(m)[:4]+'x+'+str(c)[:4])
ax[1][2].text(1,1.4,r'R$^2$ (LR) = '+str(np.min(np.corrcoef(np.log(OS),woodsed_ratio_LR))**2)[:6], color='r')

ax[1][2].set_ylabel("Ratio of normalized wood\n and normalized sediment")
ax[1][2].set_xlabel('Total sediment discharge (log tonnes)')
plt.legend()
ax[1][2].set_title('f) ', loc='left')


# plt.show()

plt.savefig("summaries/flow_sed_2011_2016_wood_sed_rel.png", dpi=300, bbox_inches="tight")
plt.close()





# plt.subplot(222)
# y = F_MR.copy()/np.nansum(F_MR,axis=0)/len(times)
# x = -.5
# r_v = (y*bins[1:]**x) / np.nansum(y*bins[1:]**x) #volume-by-weight proportion
# mnsz = np.nansum(r_v * bins[1:],axis=1)
# plt.semilogx(mnsz, dt, 'w-o')

# sig=[]
# for counter in range(len(mnsz)):
#     sig.append(np.sqrt(np.nansum(y[counter,:]*((bins[1:]-mnsz[counter])**2))))

# sig = np.array(sig)
# plt.plot(sig, dt, 'k--')
# plt.gca().invert_yaxis()
# plt.xlim(5,625)

# plt.subplot(224)
# y = F_LR.copy()/np.nansum(F_LR,axis=0)/len(times)
# mnsz = np.nansum(y * bins[1:],axis=1)
# plt.semilogx(mnsz, dt, 'r-s')

# sig=[]
# for counter in range(len(mnsz)):
#     sig.append(np.sqrt(np.nansum(y[counter,:]*((bins[1:]-mnsz[counter])**2))))

# sig = np.array(sig)
# plt.plot(sig, dt, 'r--')

# plt.gca().invert_yaxis()
# plt.xlim(5,625)




# ######################################################

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


######################################################

# MR_BR_small=[]
# MR_BR_large=[]
# for time in times:
#     tmp = MRwood_geotiffs_ds.wood.sel(time=time)
#     for g in tqdm(MRbudget_reaches_redo):
#         wood_c = tmp.rio.clip([g], tmp.rio.crs)
#         label_img = label(wood_c==1)
#         props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#         a = props['area'][np.where(props['area']<100000)[0]]
#         result2 = np.sum(a[np.where(a>4096)[0]])
#         result1 = np.sum(a[np.where(a<=4096)[0]])
#         MR_BR_small.append(float(result1))
#         MR_BR_large.append(float(result2))

# #######################################################

# LR_BR_small=[]
# LR_BR_large=[]
# for time in times:
#     tmp = LRwood_geotiffs_ds.wood.sel(time=time)
#     for g in tqdm(LRbudget_reaches_redo):
#         wood_c = tmp.rio.clip([g], tmp.rio.crs)
#         label_img = label(wood_c==1)
#         props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#         a = props['area'][np.where(props['area']<100000)[0]]
#         result2 = np.sum(a[np.where(a>4096)[0]])
#         result1 = np.sum(a[np.where(a<=4096)[0]])
#         LR_BR_small.append(float(result1))
#         LR_BR_large.append(float(result2))



# LR_BRarr_small = np.vstack(LR_BR_small).reshape(len(times),-1)/grid2sqm
# MR_BRarr_small = np.vstack(MR_BR_small).reshape(len(times),-1)/grid2sqm

# LR_BRarr_large = np.vstack(LR_BR_large).reshape(len(times),-1)/grid2sqm
# MR_BRarr_large = np.vstack(MR_BR_large).reshape(len(times),-1)/grid2sqm

# np.savez('summaries/Wood_time_series_largepieces_smallpieces.npz', LR_BRarr_large = LR_BRarr_large, MR_BRarr_large = MR_BRarr_large, LR_BRarr_small = LR_BRarr_small, MR_BRarr_small = MR_BRarr_small, times=times, dt=dt, grid2sqm=grid2sqm)



# ######################################################

# MR_BRsed=[]
# for time in times:
#     tmp1 = MRsed_geotiffs_ds.sed.sel(time=time)
#     tmp2 = MRwood_geotiffs_ds.wood.sel(time=time)
#     for g in tqdm(MRbudget_reaches_redo):
#         sed_c = tmp1.rio.clip([g], tmp1.rio.crs)
#         wood_c = tmp2.rio.clip([g], tmp2.rio.crs)

#         result = (sed_c).sum().compute().to_numpy() + (wood_c).sum().compute().to_numpy() 
#         MR_BRsed.append(float(result))
#         del wood_c, sed_c
#     del tmp1, tmp2 

# #######################################################

# LR_BRsed=[]
# for time in times:
#     tmp1 = LRsed_geotiffs_ds.sed.sel(time=time)
#     tmp2 = LRwood_geotiffs_ds.wood.sel(time=time)
#     for g in tqdm(LRbudget_reaches_redo):
#         sed_c = tmp1.rio.clip([g], tmp1.rio.crs)
#         wood_c = tmp2.rio.clip([g], tmp2.rio.crs)

#         result = (sed_c).sum().compute().to_numpy() + (wood_c).sum().compute().to_numpy() 
#         LR_BRsed.append(float(result))
#         del wood_c, sed_c
#     del tmp1, tmp2 

# LR_BRarrsed = np.vstack(LR_BRsed).reshape(len(times),-1)/grid2sqm
# MR_BRarrsed = np.vstack(MR_BRsed).reshape(len(times),-1)/grid2sqm

# np.savez('summaries/Sed_time_series.npz', LR_BRarrsed = LR_BRarrsed, MR_BRarrsed = MR_BRarrsed, times=times, dt=dt, grid2sqm=grid2sqm)



# def xcorr_spacetime(dat, tmp):
#     dat = np.flipud(dat)
#     alongcorr=[]
#     for k in np.arange(dat.shape[1]):
#         alongcorr.append(np.correlate(tmp,dat[:,k],'same'))
#     return np.vstack(alongcorr)


# cMR_BRarr = xcorr_spacetime(MR_BRarr, MR_BRarr[:,22])
# cLR_BRarr = xcorr_spacetime(LR_BRarr, LR_BRarr[:,26])
# cMR_BRarrsed = xcorr_spacetime(MR_BRarr+MR_BRarrsed, MR_BRarr[:,22]+MR_BRarrsed[:,22])
# cLR_BRarrsed = xcorr_spacetime(LR_BRarr+LR_BRarrsed, LR_BRarr[:,26]+LR_BRarrsed[:,26])


# ########################################
# plt.figure(figsize=(14,20))
# plt.subplots_adjust(wspace=0.2, hspace=0.2)

# plt.subplot(421)
# plt.imshow(cMR_BRarr/np.max(cMR_BRarr), cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=1)
# cb=plt.colorbar(); cb.set_label("Cross-correlation\n coefficient")
# plt.gca().invert_yaxis()
# plt.title('a)', loc='left')
# # plt.xlabel("Distance downstream (km)"); 

# plt.subplot(422)
# plt.imshow(cLR_BRarr/np.max(cLR_BRarr), cmap='inferno', extent=[LR[0] , LR[-1], dt[0], dt[-1]], aspect='auto', vmin=0, vmax=1)
# cb=plt.colorbar(); cb.set_label("Cross-correlation\n coefficient")
# plt.gca().invert_yaxis()
# # plt.title('c) LR', loc='left')
# # plt.xlabel("Distance downstream (km)"); 

# plt.subplot(423)
# plt.imshow(cMR_BRarrsed/np.max(cMR_BRarrsed), cmap='inferno', extent=[LR[0] , LR[-1], dt[0], dt[-1]], aspect='auto', vmin=0, vmax=1)
# cb=plt.colorbar(); cb.set_label("Cross-correlation\n coefficient")
# plt.gca().invert_yaxis()
# plt.title('b)', loc='left')
# # plt.xlabel("Distance downstream (km)"); 

# plt.subplot(424)
# plt.imshow(cLR_BRarrsed/np.max(cLR_BRarrsed), cmap='inferno', extent=[LR[0] , LR[-1], dt[0], dt[-1]], aspect='auto', vmin=0, vmax=1)
# cb=plt.colorbar(); cb.set_label("Cross-correlation\n coefficient")
# plt.gca().invert_yaxis()
# # plt.title('d) LR', loc='left')
# # plt.xlabel("Distance downstream (km)"); 

# # plt.subplot(425)
# # plt.imshow(np.flipud(MR_BRarr_c), cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=1)
# # cb=plt.colorbar(); cb.set_label("Normalized wood area\n" r"m$^2$/m$^2$")
# # plt.gca().invert_yaxis()
# # plt.title('c)', loc='left')
# # # plt.xlabel("Distance downstream (km)"); 

# # plt.subplot(427)
# # plt.imshow(np.flipud(MR_BRarrsed_c), cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=1)
# # cb=plt.colorbar(); cb.set_label("Normalized sediment area\n" r"m$^2$/m$^2$")
# # plt.gca().invert_yaxis()
# # # plt.title('g) MR', loc='left')
# # plt.xlabel("River kilometer"); 

# # plt.subplot(426)
# # plt.imshow(np.flipud(LR_BRarr_c), cmap='inferno', extent=[LR[0], LR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=1)
# # cb=plt.colorbar(); cb.set_label("Normalized wood area\n" r"m$^2$/m$^2$")
# # plt.gca().invert_yaxis()
# # plt.title('d)', loc='left')
# # # plt.xlabel("Distance downstream (km)"); 

# # plt.subplot(428)
# # plt.imshow(np.flipud(LR_BRarrsed_c), cmap='inferno', extent=[LR[0], LR[-1] , dt[0], dt[-1]], aspect='auto', vmin=0, vmax=1)
# # cb=plt.colorbar(); cb.set_label("Normalized sediment area\n" r"m$^2$/m$^2$")
# # plt.gca().invert_yaxis()
# # # plt.title('d)', loc='left');
# # plt.xlabel("River kilometer"); 

# plt.show()





# MRdx, MRdt = np.gradient(np.flipud(MR_BRarr))
# LRdx, LRdt = np.gradient(np.flipud(LR_BRarr))

# sMRdx, sMRdt = np.gradient(np.flipud(MR_BRarrsed+MR_BRarr))
# sLRdx, sLRdt = np.gradient(np.flipud(LR_BRarrsed+LR_BRarr))


# ########################################
# plt.figure(figsize=(14,20))
# plt.subplots_adjust(wspace=0.2, hspace=0.2)

# plt.subplot(421)
# plt.imshow(MRdx, cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"$\Delta x$ Wood area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('a)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

# plt.subplot(422)
# plt.imshow(MRdt, cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"$\Delta t$ Wood area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('b)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

# plt.subplot(423)
# plt.imshow(LRdx, cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"$\Delta x$ Wood area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('c)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

# plt.subplot(424)
# plt.imshow(LRdt, cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"$\Delta t$ Wood area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('d)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

# plt.subplot(425)
# plt.imshow(sMRdx, cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"$\Delta x$ Sediment area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('a)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

# plt.subplot(426)
# plt.imshow(sMRdt, cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"$\Delta t$ Sediment area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('b)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

# plt.subplot(427)
# plt.imshow(sLRdx, cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"$\Delta x$ Sediment area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('c)', loc='left')
# plt.xlabel("Distance downstream (km)"); 

# plt.subplot(428)
# plt.imshow(sLRdt, cmap='inferno', extent=[MR[0], MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"$\Delta t$ Sediment area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('d)', loc='left')
# plt.xlabel("Distance downstream (km)"); 
# plt.show()





# ####### sed
# # plt.subplot(221)
# f,bins=np.histogram(sed_load['Total sediment discharge (tonnes)'][ind].values)
# ax[1][2].plot(bins[:-1],f/f.max(),'k')
# data = np.hstack([np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values)]*2)
# data = np.hstack((data, np.random.randint(data.min(), data.max(),len(data))))
# f,bins=np.histogram(data)
# ax[1][2].bar(bins[:-1],f/f.max(),width=200,color='r')
# ax[1][2].set_ylabel(r"Normalized frequency")

# ax[1][2].set_xlabel('Total sediment discharge (tonnes)', color='k')
# ax[1][2].set_title('f) ', loc='left')
# ax[1][2].set_xlim((0,200000))


# fig, ax1 = plt.subplots()

# ax1.plot(dt, np.sum(MR_BRarr,axis=1),'k-')
# ax1.plot(dt, np.sum(LR_BRarr,axis=1),'r--')
# ax1.set_ylabel('Total wood area (m2)')


# ax2 = ax1.twinx()
# ax2.plot(dt_sed[ind], sed_load['Daily Discharge (m3/s)'][ind],'b-')
# ax2.set_ylabel('Daily Discharge (m3/s)', color='k')

# # plt.show()
# plt.savefig("summaries/flow_versus_wood.png", dpi=300, bbox_inches="tight")
# plt.close()


# ########################################
# plt.figure(figsize=(8,8))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(311)
# plt.plot(dt_sed[ind], sed_load['Daily Discharge (m3/s)'][ind],'k-')
# plt.ylabel(r'Discharge (m$^3$/s)')
# plt.title('a) ', loc='left')

# plt.subplot(312)
# plt.plot(dt_sed[ind], sed_load['Total sediment discharge (tonnes)'][ind],'k-')
# plt.ylabel(r'Total sediment discharge (tonnes)')
# plt.title('b) ', loc='left')

# plt.subplot(313)
# plt.plot(dt_sed[ind], sed_load['Ave fraction fines (based on two turbidimeters)'][ind],'k-')
# plt.ylabel(r'Average fraction of fines')
# plt.title('c) ', loc='left')

# # plt.show()
# plt.savefig("flow_sed_2011_2016.png", dpi=300, bbox_inches="tight")
# plt.close()


# ########################################

# fig, ax = plt.subplots(nrows=2, ncols=3)
# fig.set_size_inches(16,12)
# plt.subplots_adjust(wspace=0.6, hspace=0.2)

# ####### flow
# # plt.subplot(221)
# ax[0][0].plot(dt, np.sum(MR_BRarr,axis=1),'k-')
# ax[0][0].plot(dt, np.sum(LR_BRarr,axis=1),'r--')
# ax[0][0].set_ylabel('Total wood area (m2)')

# ax2 = ax[0][0].twinx()
# ax2.plot(dt_sed[ind], sed_load['Daily Discharge (m3/s)'][ind],'b-')
# ax2.set_ylabel(r'Daily Discharge (m$^3$/s)', color='k')
# ax[0][0].set_title('a) ', loc='left')

# # ax[0][0].plot(dt, np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values), 'go')

# # plt.subplot(222)
# ax[0][1].plot(np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values), np.sum(MR_BRarr,axis=1), 'bo')
# ax[0][1].plot(np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values), np.sum(LR_BRarr,axis=1), 'rs')

# O = np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values)
# E = np.sum(LR_BRarr,axis=1)

# A = np.vstack([O, np.ones(len(O))]).T
# m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
# ax[0][1].plot(O, m*O+ c, 'r:',lw=2, label='(LR) y = '+str(m)[:4]+'x+'+str(c)[:4])
# ax[0][1].text(20,45000,r'R$^2$ = '+str(np.min(np.corrcoef(O,E))**2)[:6], color='r')

# E = np.sum(MR_BRarr,axis=1)
# m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
# ax[0][1].plot(O, m*O+ c, 'b:',lw=2, label='(MR) y = '+str(m)[:4]+'x+'+str(c)[:4])
# ax[0][1].text(20,40000,r'R$^2$ = '+str(np.min(np.corrcoef(O,E))**2)[:6], color='b')

# ax[0][1].legend(fontsize=7)
# ax[0][1].set_ylabel(r"Estimated wood, m$^2$")
# ax[0][1].set_xlabel(r'Discharge, day of aerial survey  (m$^3$/s)')
# ax[0][1].set_title('c) ', loc='left')


# ########### sediment
# # plt.subplot(223)
# ax[1][0].plot(dt, np.sum(MR_BRarr,axis=1),'k-')
# ax[1][0].plot(dt, np.sum(LR_BRarr,axis=1),'r--')
# ax[1][0].set_ylabel('Total wood area (m2)')

# ax2 = ax[1][0].twinx()
# ax2.plot(dt_sed[ind], sed_load['Total sediment discharge (tonnes)'][ind],'b-')
# ax2.set_ylabel('Total sediment discharge (tonnes)', color='k')
# ax[1][0].set_title('b) ', loc='left')

# # ax[1][0].plot(dt, np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values), 'go')


# # plt.subplot(224)
# ax[1][1].plot(np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values), np.sum(MR_BRarr,axis=1), 'bo')
# ax[1][1].plot(np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values), np.sum(LR_BRarr,axis=1), 'rs')

# O = np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values)
# E = np.sum(LR_BRarr,axis=1)

# A = np.vstack([O, np.ones(len(O))]).T
# m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
# ax[1][1].plot(O, m*O+ c, 'r:',lw=2, label='(LR) y = '+str(m)[:4]+'x+'+str(c)[:4])
# ax[1][1].text(500,45000,r'R$^2$ = '+str(np.min(np.corrcoef(O,E))**2)[:6], color='r')

# E = np.sum(MR_BRarr,axis=1)
# m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
# ax[1][1].plot(O, m*O+ c, 'b:',lw=2, label='(MR) y = '+str(m)[:4]+'x+'+str(c)[:4])
# ax[1][1].text(500,40000,r'R$^2$ = '+str(np.min(np.corrcoef(O,E))**2)[:6], color='b')

# ax[1][1].legend(fontsize=7)
# ax[1][1].set_ylabel(r"Estimated wood, m$^2$")
# ax[1][1].set_xlabel(r'Total sediment discharge, day of aerial survey (tonnes)')
# ax[1][1].set_title('d) ', loc='left')


# ####### flow
# # plt.subplot(221)
# f,bins=np.histogram(sed_load['Daily Discharge (m3/s)'][ind].values)
# ax[0][2].plot(bins[:-1],f/f.max(),'k')
# data = np.hstack([np.interp(t, t_sed, sed_load['Daily Discharge (m3/s)'][ind].values)]*2)
# data = np.hstack((data, np.random.randint(data.min(), data.max(),len(data))))
# f,bins=np.histogram(data)
# ax[0][2].bar(bins[:-1],f/f.max(),width=2,color='r')
# # ax[0][2].set_ylabel('Total wood area (m2)')

# ax[0][2].set_ylabel(r"Normalized frequency")
# ax[0][2].set_xlabel(r'Daily Discharge (m$^3$/s)', color='k')
# ax[0][2].set_title('e) ', loc='left')


# ####### sed
# # plt.subplot(221)
# f,bins=np.histogram(sed_load['Total sediment discharge (tonnes)'][ind].values)
# ax[1][2].plot(bins[:-1],f/f.max(),'k')
# data = np.hstack([np.interp(t, t_sed, sed_load['Total sediment discharge (tonnes)'][ind].values)]*2)
# data = np.hstack((data, np.random.randint(data.min(), data.max(),len(data))))
# f,bins=np.histogram(data)
# ax[1][2].bar(bins[:-1],f/f.max(),width=200,color='r')
# ax[1][2].set_ylabel(r"Normalized frequency")

# ax[1][2].set_xlabel('Total sediment discharge (tonnes)', color='k')
# ax[1][2].set_title('f) ', loc='left')
# ax[1][2].set_xlim((0,200000))

# plt.savefig("summaries/flow_sed_2011_2016_wood_rel.png", dpi=300, bbox_inches="tight")
# plt.close()



# plt.figure(figsize=(24,4))
# plt.subplots_adjust(wspace=0.5, hspace=0.3)

# plt.subplot(151)
# plt.plot(dt_sed[ind], sed_load['Daily Discharge (m3/s)'][ind],'k-')
# plt.ylabel(r'Discharge (m$^3$/s)')
# plt.title('a) ', loc='left')

# plt.subplot(152)
# plt.plot(dt_sed[ind], sed_load['Total sediment discharge (tonnes)'][ind],'k-')
# plt.ylabel(r'Total sediment discharge (tonnes)')
# plt.title('b) ', loc='left')

# plt.subplot(153)
# plt.plot(dt_sed[ind], sed_load['Ave fraction fines (based on two turbidimeters)'][ind],'k-')
# plt.ylabel(r'Average fraction of fines')
# plt.title('c) ', loc='left')

# plt.subplot(154)
# plt.plot(dt,np.sum(MR_BRarr,axis=1)/np.sum(A_MR),'k-', label='MR')
# plt.plot(dt,np.sum(LR_BRarr,axis=1)/np.sum(A_MR),'r--', label='LR')
# plt.ylabel(r"Estimated wood, m$^2$")
# plt.title('d) ', loc='left')







# plt.subplot(323)
# plt.imshow(np.flipud(np.cumsum(MR_BRarr.T,axis=0).T), cmap='inferno', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Cumulative wood area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('c) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(324)
# plt.imshow(np.flipud(np.cumsum(LR_BRarr.T,axis=0).T), cmap='inferno', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Cumulative wood area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('d) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(325)
# plt.plot(MR, np.sum(MR_BRarr,axis=0),'k-', label='MR')
# plt.plot(LR, np.sum(LR_BRarr,axis=0),'r--', label='LR')
# plt.ylabel(r"Sum of estimated wood, m$^2$"); plt.xlabel("Distance downstream (km)"); 
# plt.legend()
# plt.title('e) ', loc='left')

# plt.subplot(326)
# plt.plot(dt[summer],np.sum(MR_BRarr,axis=1)[summer],'k-o', label='MR, Discharge < 30 m$^3$/s')
# plt.plot(dt[summer],np.sum(LR_BRarr,axis=1)[summer],'r--o', label='LR, Discharge < 30 m$^3$/s')

# plt.plot(dt[winter],np.sum(MR_BRarr,axis=1)[winter],'k-*', label='MR, Discharge > 30 m$^3$/s')
# plt.plot(dt[winter],np.sum(LR_BRarr,axis=1)[winter],'r--*', label='LR, Discharge > 30 m$^3$/s')

# # plt.plot(dt,np.sum(MR_BRarr,axis=1)-np.sum(MR_BRarr[0,:]),'b-', label='MR, rel. start')
# # plt.plot(dt,np.sum(LR_BRarr,axis=1)-np.sum(LR_BRarr[0,:]),'b--', label='LR, rel. start')

# plt.ylabel(r"Sum of estimated wood, m$^2$");
# plt.legend()
# plt.plot(dt,np.sum(MR_BRarr,axis=1),'-',color=[0.75,0.75,0.75],lw=2, alpha=0.5)
# plt.plot(dt,np.sum(LR_BRarr,axis=1),'--', color=[0.25,0.25,0.25],lw=2, alpha=0.5)
# plt.title('f) ', loc='left')
# # plt.show()

# plt.savefig("summaries/wood_spacetime_plots.png", dpi=300, bbox_inches="tight")
# plt.close()





########################################
# plt.figure(figsize=(16,16))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(321)
# plt.imshow(np.flipud(MR_BRarrsed), cmap='inferno', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Sediment area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('a) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(322)
# plt.imshow(np.flipud(LR_BRarrsed), cmap='inferno', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Sediment area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(323)
# plt.imshow(np.flipud(np.cumsum(MR_BRarrsed.T,axis=0).T), cmap='inferno', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Cumulative sediment area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('c) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(324)
# plt.imshow(np.flipud(np.cumsum(LR_BRarrsed.T,axis=0).T), cmap='inferno', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Cumulative sediment area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('d) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(325)
# plt.plot(MR, np.sum(MR_BRarrsed,axis=0),'k-', label='MR')
# plt.plot(LR, np.sum(LR_BRarrsed,axis=0),'r--', label='LR')
# plt.ylabel(r"Sum of estimated sediment, m$^2$"); plt.xlabel("Distance downstream (km)"); 
# plt.legend()
# plt.title('e) ', loc='left')

# plt.subplot(326)
# # plt.plot(dt,np.sum(MR_BRarrsed,axis=1),'k-', label='MR')
# # plt.plot(dt,np.sum(LR_BRarrsed,axis=1),'r--', label='LR')
# plt.plot(dt[summer],np.sum(MR_BRarrsed,axis=1)[summer],'k-o', label='MR, Discharge < 30 m$^3$/s')
# plt.plot(dt[summer],np.sum(LR_BRarrsed,axis=1)[summer],'r--o', label='LR, Discharge < 30 m$^3$/s')

# plt.plot(dt[winter],np.sum(MR_BRarrsed,axis=1)[winter],'k-*', label='MR, Discharge > 30 m$^3$/s')
# plt.plot(dt[winter],np.sum(LR_BRarrsed,axis=1)[winter],'r--*', label='LR, Discharge > 30 m$^3$/s')

# # plt.plot(dt,np.sum(MR_BRarrsed,axis=1)-np.sum(MR_BRarrsed[0,:]),'b-', label='MR, rel. start')
# # plt.plot(dt,np.sum(LR_BRarrsed,axis=1)-np.sum(LR_BRarrsed[0,:]),'b--', label='LR, rel. start')

# plt.ylabel(r"Sum of estimated sediment, m$^2$");
# plt.legend()
# plt.plot(dt,np.sum(MR_BRarrsed,axis=1),'-',color=[0.75,0.75,0.75],lw=2, alpha=0.5)
# plt.plot(dt,np.sum(LR_BRarrsed,axis=1),'--', color=[0.25,0.25,0.25],lw=2, alpha=0.5)
# plt.title('f) ', loc='left')
# # plt.show()

# plt.savefig("summaries/sediment_spacetime_plots.png", dpi=300, bbox_inches="tight")
# plt.close()






# plt.plot(dt[1:],np.cumsum(np.diff(np.sum(MR_BRarr,axis=1)-np.sum(MR_BRarr[0,:]))),'k-', label='MR')
# plt.plot(dt[1:],np.cumsum(np.diff(np.sum(LR_BRarr,axis=1)-np.sum(LR_BRarr[0,:]))),'r--', label='LR')

# A = []
# for k in range(MR_BRarr.shape[0]):
#     A.append(np.correlate(MR_BRarr[0,:],MR_BRarr[k,:],'full'))

# ########################################
# plt.figure(figsize=(16,16))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(321)
# plt.imshow(np.vstack(A), cmap='inferno', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Wood area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('a) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.show()


# plt.subplot(321)
# plt.imshow(MR_BRarr, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Wood area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('a) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(322)
# plt.imshow(LR_BRarr, cmap='viridis', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
# cb=plt.colorbar(); cb.set_label(r"Wood area, m$^2$")
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel("Distance downstream (km)"); 


# ########################################
# plt.figure(figsize=(16,16))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(321)
# plt.imshow(MR_BRarr_large, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('a) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(322)
# plt.imshow(LR_BRarr_large, cmap='viridis', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(323)
# plt.imshow(np.cumsum(MR_BRarr_large.T,axis=0).T, cmap='viridis', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('c) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(324)
# plt.imshow(np.cumsum(LR_BRarr_large.T,axis=0).T, cmap='viridis', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('d) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(325)
# plt.plot(MR, np.sum(MR_BRarr_large,axis=0),'k-', label='MR')
# plt.plot(LR, np.sum(LR_BRarr_large,axis=0),'r--', label='LR')
# plt.ylabel(r"Sum of estimated wood, m$^2$"); plt.xlabel("Distance downstream (km)"); 
# plt.legend()
# plt.title('e) ', loc='left')

# plt.subplot(326)
# plt.plot(dt,np.sum(MR_BRarr_large,axis=1),'k-', label='MR')
# plt.plot(dt,np.sum(LR_BRarr_large,axis=1),'r--', label='LR')
# plt.ylabel(r"Sum of estimated wood, m$^2$");
# plt.legend()
# plt.title('f) ', loc='left')
# # plt.show()

# plt.savefig("wood_spacetime_plots_largewood_only.png", dpi=300, bbox_inches="tight")
# plt.close()



# MR_BRarr_c = MR_BRarr_large/A_MR

# LR_BRarr_c = LR_BRarr_large/A_LR

# ########################################
# plt.figure(figsize=(16,16))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

# plt.subplot(321)
# plt.imshow(np.flipud(MR_BRarr_c), cmap='inferno', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('a) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(322)
# plt.imshow(np.flipud(LR_BRarr_c), cmap='inferno', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('b) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(323)
# plt.imshow(np.flipud(np.cumsum(MR_BRarr_c.T,axis=0).T), cmap='inferno', extent=[0, MR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('c) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(324)
# plt.imshow(np.flipud(np.cumsum(LR_BRarr_c.T,axis=0).T), cmap='inferno', extent=[0, LR[-1] , dt[0], dt[-1]], aspect='auto')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('d) ', loc='left'); plt.xlabel("Distance downstream (km)"); 

# plt.subplot(325)
# plt.plot(MR, np.sum(MR_BRarr,axis=0)/(A_MR*len(times)),'k-', label='MR')
# plt.plot(LR, np.sum(LR_BRarr,axis=0)/(A_LR*len(times)),'r--', label='LR')
# plt.ylabel(r"Wood concentration, m$^2$/m$^2$"); plt.xlabel("Distance downstream (km)"); 
# plt.legend()
# plt.title('e) ', loc='left')

# plt.subplot(326)

# plt.plot(dt[summer],np.sum(MR_BRarr,axis=1)[summer]/np.sum(A_MR),'k-o', label='MR, Discharge < 30 m$^3$/s')
# plt.plot(dt[summer],np.sum(LR_BRarr,axis=1)[summer]/np.sum(A_LR),'r--o', label='LR, Discharge < 30 m$^3$/s')

# plt.plot(dt[winter],np.sum(MR_BRarr,axis=1)[winter]/np.sum(A_MR),'k-*', label='MR, Discharge > 30 m$^3$/s')
# plt.plot(dt[winter],np.sum(LR_BRarr,axis=1)[winter]/np.sum(A_LR),'r--*', label='LR, Discharge > 30 m$^3$/s')

# plt.ylabel(r"Wood concentration, m$^2$/m$^2$");
# plt.legend()
# plt.plot(dt,np.sum(MR_BRarr,axis=1)/np.sum(A_MR),'-',color=[0.75,0.75,0.75],lw=2, alpha=0.5)
# plt.plot(dt,np.sum(LR_BRarr,axis=1)/np.sum(A_LR),'--', color=[0.25,0.25,0.25],lw=2, alpha=0.5)

# plt.title('f) ', loc='left')
# # plt.show()

# plt.savefig("summaries/wood_conc_spacetime_plots.png", dpi=300, bbox_inches="tight")
# plt.close()





#### plot size-frequency distribtuions in time



#### autocorrelation




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
