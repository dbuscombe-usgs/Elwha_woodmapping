

## Dan Buscombe, Marda Science


import json, os
import rioxarray
import xarray as xr 
from glob import glob 
from dask.distributed import Client
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
# from scipy import ndimage
# from skimage.exposure import match_histograms
import pandas as pd 
# import mchmm as mc
from datetime import datetime
from area import area
import geopandas as gpd
from scipy.ndimage import uniform_filter1d
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
## start client
# client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

cwd = os.getcwd()

# Create variable used for time axis
time_var = xr.Variable('time',times)

dt = [datetime.strptime(time,'%Y-%m-%d') for time in times]

## budget reaches
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

A_LR = np.array(A_LR)
A_MR = np.array(A_MR)


### active widths 
file = '../raw_data/GIS/LR_active_widths.geojson'
LR_widths = gpd.read_file(file)
LR_widths = LR_widths['length'].values

file = '../raw_data/GIS/MR_active_widths.geojson'
MR_widths = gpd.read_file(file)
MR_widths = MR_widths['length'].values


### distances downstream 
dists = pd.read_csv('br_dists.csv')
LR = np.hstack((0,np.array(dists['LR'])))
MR = np.hstack((0,np.array(dists['MR'][:43])))
## rescale distances
LR = rescale_array(LR,11,2)
MR = rescale_array(MR[::-1],12,20)


### elev, MR
fpoints = sorted(glob('../raw_data/GIS/Sep22_2017/MR_dem_pts_braids_epsg6339.geojson'))
with open(fpoints[0]) as f:
    gj = json.load(f)
features = gj['features']

points = [f['geometry']['coordinates'] for f in features]
z = [f['properties']['Elwha_MR_20170922_DEM_regrid_1'] for f in features]
print("{} sample points".format(len(z)))

zMR=sorted(z)

### elev, LR
fpoints = sorted(glob('../raw_data/GIS/Sep22_2017/dem_pts_braids.geojson'))
with open(fpoints[0]) as f:
    gj = json.load(f)
features = gj['features']

points = [f['geometry']['coordinates'] for f in features]
z = [f['properties']['Elwha_LR_20170922_DEM_regrid_1'] for f in features]
print("{} sample points".format(len(z)))

zLR=sorted(z)


################ both reaches
zLR = np.array(zLR)
zMR = np.array(zMR)



####################################################


with np.load('summaries/Sed_time_series.npz', allow_pickle=True) as f:
    LR_BRarrsed = f['LR_BRarrsed']
    MR_BRarrsed = f['MR_BRarrsed']



with np.load('summaries/Wood_time_series.npz', allow_pickle=True) as f:
    LR_BRarr = f['LR_BRarr']
    MR_BRarr = f['MR_BRarr']
    dt = f['dt']
    grid2sqm = f['grid2sqm']


MR_wood = np.sum(MR_BRarr,axis=0)
MR_sed = np.sum(MR_BRarr,axis=0)+np.sum(MR_BRarrsed,axis=0)

LR_wood = np.sum(LR_BRarr,axis=0)
LR_sed = np.sum(LR_BRarr,axis=0)+np.sum(LR_BRarrsed,axis=0)


MR_wood_c = MR_wood/A_MR/len(times)
MR_sed_c = MR_sed/A_MR/len(times)
LR_wood_c = LR_wood/A_LR/len(times)
LR_sed_c = LR_sed/A_LR/len(times)

############# TPMS

with np.load('summaries/LR_transition_matrices.npz', allow_pickle=True) as dat:
    LR_tpm = dict()
    for k in dat.keys():
        LR_tpm[k] = dat[k]
    del dat

with np.load('summaries/MR_transition_matrices.npz', allow_pickle=True) as dat:
    MR_tpm = dict()
    for k in dat.keys():
        MR_tpm[k] = dat[k]
    del dat

MR_TPM = []
for k in MR_tpm['MR_PM']:
    if np.isnan(k).any():
        tmp = np.ones((4,4))*np.nan
        MR_TPM.append(tmp)
    else:
        MR_TPM.append(k)

LR_TPM = []
for k in LR_tpm['LR_PM']:
    if np.isnan(k).any():
        tmp = np.ones((4,4))*np.nan
        LR_TPM.append(tmp)
    else:
        LR_TPM.append(k)

### resample all vectors to common extent
xMRi = np.linspace(0,len(MR),len(zMR))
zMRi = rescale_array(np.interp(xMRi, np.linspace(0,len(zMR),len(xMRi)), zMR),np.min(zMR), np.max(zMR))
gzMRi = np.gradient(zMRi)


zLRc = zLR[134:]
xLRi = np.linspace(0,len(LR),len(zLRc))
zLRi = rescale_array(np.interp(xLRi, np.linspace(0,len(zLRc),len(xLRi)), zLRc),np.min(zLRc), np.max(zLRc))
gzLRi = np.gradient(zLRi)


# plt.plot(xMRi,zMRi,'k')
# plt.plot(xLRi,zLRi,'r')
# plt.show()

## resample widths
xMRi = np.linspace(0,len(MR),len(MR_widths))
wMRi = np.interp(xMRi, np.linspace(0,len(MR_widths),len(xMRi)), MR_widths)

xLRi = np.linspace(0,len(LR),len(LR_widths))
wLRi = np.interp(xLRi, np.linspace(0,len(LR_widths),len(xLRi)), LR_widths)

## resample wood loads
xMRi = np.linspace(0,len(MR),len(MR_widths))
woodMRi = np.interp(xMRi, np.linspace(0,len(xMRi),len(MR_wood)), MR_wood)

xLRi = np.linspace(0,len(LR),len(LR_widths))
woodLRi = np.interp(xLRi, np.linspace(0,len(xLRi),len(LR_wood)), LR_wood)

## resample sed loads
xMRi = np.linspace(0,len(MR),len(MR_widths))
sedMRi = np.interp(xMRi, np.linspace(0,len(xMRi),len(MR_sed)), MR_sed)

xLRi = np.linspace(0,len(LR),len(LR_widths))
sedLRi = np.interp(xLRi, np.linspace(0,len(xLRi),len(LR_sed)), LR_sed)


## resample wood concs
xMRi = np.linspace(0,len(MR),len(MR_widths))
c_woodMRi = np.interp(xMRi, np.linspace(0,len(xMRi),len(MR_wood)), MR_wood_c)

xLRi = np.linspace(0,len(LR),len(LR_widths))
c_woodLRi = np.interp(xLRi, np.linspace(0,len(xLRi),len(LR_wood)), LR_wood_c)

## resample sed concs
xMRi = np.linspace(0,len(MR),len(MR_widths))
c_sedMRi = np.interp(xMRi, np.linspace(0,len(xMRi),len(MR_sed)), MR_sed_c)

xLRi = np.linspace(0,len(LR),len(LR_widths))
c_sedLRi = np.interp(xLRi, np.linspace(0,len(xLRi),len(LR_sed)), LR_sed_c)


## resample elev

gzMRi = gzMRi[gzMRi>0]
gzLRi = gzLRi[gzLRi>0]

xMRi = np.linspace(0,len(MR),len(MR_widths))
c_gzMRi= np.interp(xMRi, np.linspace(0,len(xMRi),len(gzMRi)), gzMRi)

xLRi = np.linspace(0,len(LR),len(LR_widths))
c_gzLRi= np.interp(xLRi, np.linspace(0,len(xLRi),len(gzLRi)), gzLRi)


# Cm = []
# Cl = []
# for s in np.linspace(1,10,10):

#     wMRic = uniform_filter1d(wMRi, size=int(s))
#     wLRic = uniform_filter1d(wLRi, size=int(s))

#     Cm.append(np.min(np.corrcoef(wMRic,woodMRi)))
#     Cl.append(np.min(np.corrcoef(wLRic,woodLRi)))

# plt.plot(np.linspace(1,10,10), Cm,'k')
# plt.plot(np.linspace(1,10,10), Cl,'r')
# plt.show()



wMRic = uniform_filter1d(wMRi, size=6)
wLRic = uniform_filter1d(wLRi, size=6)





plt.figure(figsize=(14,18))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(321)
plt.plot(wMRic,woodMRi,'ko',label='MR',alpha=0.5)
O = np.vstack([wMRic, np.ones(len(wMRi))]).T
m, c = np.linalg.lstsq(O, woodMRi, rcond=None)[0]
plt.plot(np.sort(wMRic), m*np.sort(wMRic)+ c, 'k:',lw=2, alpha=0.5, label='(MR) y = '+str(m)[:4]+'x+'+str(c)[:4])

plt.plot(wLRic,woodLRi,'rs',label='LR')
O = np.vstack([wLRic, np.ones(len(wLRic))]).T
m, c = np.linalg.lstsq(O, woodLRi, rcond=None)[0]
plt.plot(np.sort(wLRic), m*np.sort(wLRic)+ c, 'r:',lw=2, label='(LR) y = '+str(m)[:4]+'x+'+str(c)[:4])
plt.ylim(0,35000)
plt.xlim(0,500)
plt.text(380,3800,r'R$^2$ = '+str(np.min(np.corrcoef(wMRic,woodMRi))**2)[:6], color='k')
plt.text(380,2500,r'R$^2$ = '+str(np.min(np.corrcoef(wLRic,woodLRi))**2)[:6], color='r')
plt.title('a) ', loc='left')
plt.ylabel(r"Estimated wood, m$^2$")
plt.xlabel(r"Maximum channel width (m)")
plt.legend()

plt.subplot(322)
plt.plot(wMRic,sedMRi,'ko',label='MR',alpha=0.5)
O = np.vstack([wMRic, np.ones(len(wMRic))]).T
m, c = np.linalg.lstsq(O, sedMRi, rcond=None)[0]
plt.plot(np.sort(wMRic), m*np.sort(wMRic)+ c, 'k:',lw=2, alpha=0.5, label='(MR) y = '+str(m)[:4]+'x+'+str(c)[:4])

plt.plot(wLRic,sedLRi,'rs',label='LR')
O = np.vstack([wLRic, np.ones(len(wLRi))]).T
m, c = np.linalg.lstsq(O, sedLRi, rcond=None)[0]
plt.plot(np.sort(wLRic), m*np.sort(wLRic)+ c, 'r:',lw=2, label='(LR) y = '+str(m)[:4]+'x+'+str(c)[:4])
plt.ylim(0,250000)
plt.xlim(0,500)
plt.text(380,30000,r'R$^2$ = '+str(np.min(np.corrcoef(wMRic,sedMRi))**2)[:6], color='k')
plt.text(380,20000,r'R$^2$ = '+str(np.min(np.corrcoef(wLRic,sedLRi))**2)[:6], color='r')
plt.title('b) ', loc='left')
plt.legend()
plt.ylabel(r"Estimated sediment, m$^2$")
plt.xlabel(r"Maximum channel width (m)")

plt.subplot(323)
plt.plot(wMRic,c_woodMRi,'ko',label='MR',alpha=0.5)
O = np.vstack([wMRic, np.ones(len(wMRi))]).T
m, c = np.linalg.lstsq(O, c_woodMRi, rcond=None)[0]
plt.plot(np.sort(wMRic), m*np.sort(wMRic)+ c, 'k:',lw=2, alpha=0.5, label='(MR) y = '+str(m)[:4]+'x+'+str(c)[:4])

plt.plot(wLRic,c_woodLRi,'rs',label='LR')
O = np.vstack([wLRic, np.ones(len(wLRic))]).T
m, c = np.linalg.lstsq(O, c_woodLRi, rcond=None)[0]
plt.plot(np.sort(wLRic), m*np.sort(wLRic)+ c, 'r:',lw=2, label='(LR) y = '+str(m)[:4]+'x+'+str(c)[:4])
plt.ylim(0,.1)
plt.xlim(0,500)
# plt.text(380,3800,r'R$^2$ = '+str(np.min(np.corrcoef(wMRic,c_woodMRi))**2)[:6], color='k')
# plt.text(380,2500,r'R$^2$ = '+str(np.min(np.corrcoef(wLRic,c_woodLRi))**2)[:6], color='r')
plt.title('c) ', loc='left')
plt.ylabel("Normalized wood area\n" r"m$^2$/m$^2$")
plt.xlabel(r"Maximum channel width (m)")
plt.legend()

plt.subplot(324)
plt.plot(wMRic,c_sedMRi,'ko',label='MR',alpha=0.5)
O = np.vstack([wMRic, np.ones(len(wMRic))]).T
m, c = np.linalg.lstsq(O, c_sedMRi, rcond=None)[0]
plt.plot(np.sort(wMRic), m*np.sort(wMRic)+ c, 'k:',lw=2, alpha=0.5, label='(MR) y = '+str(m)[:4]+'x+'+str(c)[:4])

plt.plot(wLRic,c_sedLRi,'rs',label='LR')

O = np.vstack([wLRic, np.ones(len(wLRi))]).T
m, c = np.linalg.lstsq(O, c_sedLRi, rcond=None)[0]
plt.plot(np.sort(wLRic), m*np.sort(wLRic)+ c, 'r:',lw=2, label='(LR) y = '+str(m)[:4]+'x+'+str(c)[:4])
plt.ylim(0,.65)
plt.xlim(0,500)
# plt.text(380,30000,r'R$^2$ = '+str(np.min(np.corrcoef(wMRic,sedMRi))**2)[:6], color='k')
# plt.text(380,20000,r'R$^2$ = '+str(np.min(np.corrcoef(wLRic,sedLRi))**2)[:6], color='r')
plt.title('d) ', loc='left')
plt.legend()
plt.ylabel("Normalized sediment area\n" r"m$^2$/m$^2$")
plt.xlabel(r"Maximum channel width (m)")


plt.subplot(325)
plt.plot(c_gzMRi,woodMRi,'ko',label='MR',alpha=0.5)
plt.plot(c_gzLRi,woodLRi,'rs',label='LR')
plt.xlim(0,0.016)
plt.ylim(0,35000)
plt.title('e) ', loc='left')
plt.ylabel(r"Estimated wood, m$^2$")
plt.xlabel(r"River gradient (m/m)")
plt.legend()

plt.subplot(326)
plt.plot(c_gzMRi,sedMRi,'ko',label='MR',alpha=0.5)
plt.plot(c_gzLRi,sedLRi,'rs',label='LR')
plt.xlim(0,0.016)
plt.ylim(0,250000)
plt.title('f) ', loc='left')
plt.ylabel(r"Estimated sediment, m$^2$")
plt.xlabel(r"River gradient (m/m)")
plt.legend()

# plt.show()
plt.savefig("summaries/sedimentwood_rel_width_grad.png", dpi=300, bbox_inches="tight")
plt.close()



# plt.subplot(233)
# plt.plot(wMRi,woodMRi/sedMRi,'ko',label='MR')
# plt.plot(wLRi,woodLRi/sedLRi,'rs',label='LR')
# # plt.legend()

# # plt.subplot(234)
# # plt.plot(wMRi,c_woodMRi,'ko',label='MR')
# # plt.plot(wLRi,c_woodLRi,'rs',label='LR')
# # # plt.legend()

# # plt.subplot(235)
# # plt.plot(wMRi,c_sedMRi,'ko',label='MR')
# # plt.plot(wLRi,c_sedLRi,'rs',label='LR')
# # # plt.legend()

# # plt.subplot(236)
# # plt.plot(wMRi,c_woodMRi/c_sedMRi,'ko',label='MR')
# # plt.plot(wLRi,c_woodLRi/c_sedLRi,'rs',label='LR')
# # # plt.legend()

# plt.show()









# Cm = []
# Cl = []
# for s in np.linspace(1,5000,5000):

#     gzLRic = uniform_filter1d(gzLRi, size=int(s))
#     gzMRic = uniform_filter1d(gzMRi, size=int(s))

#     Cm.append(np.min(np.corrcoef(wMRi,gzMRic[::311])))
#     Cl.append(np.min(np.corrcoef(wLRi,gzLRic[::277])))


# plt.plot(np.linspace(1,5000,5000), Cm,'k')
# plt.plot(np.linspace(1,5000,5000), Cl,'r')
# plt.show()


# gzLRic = uniform_filter1d(gzLRi, size=np.argmin(Cl))
# gzMRic = uniform_filter1d(gzMRi, size=np.argmin(Cm))


# plt.plot(gzMRic,'k')
# plt.plot(gzLRic,'r')
# plt.show()


# print(np.mean(gzMRic))
# print(np.mean(gzLRic))

# np.min(np.corrcoef(wMRi,gzMRic[::311]))
# np.min(np.corrcoef(wLRi,gzLRic[::277]))



# xMRi = np.linspace(0,len(wMRi),len(MR))
# wMRic = np.interp(xMRi, np.linspace(0,len(wMRi),len(wMRi)), wMRi)

# xLRi = np.linspace(0,len(wLRi),len(LR))
# wLRic = np.interp(xLRi, np.linspace(0,len(wLRi),len(wLRi)), wLRi)


# gMRic = np.interp(xMRi, np.linspace(0,len(wMRi),len(wMRi)), gzMRic[::311])
# gLRic = np.interp(xLRi, np.linspace(0,len(wLRi),len(wLRi)), gzLRic[::277])


# plt.plot(MR,wMRic,'k')
# plt.plot(LR,wLRic,'r')
# plt.show()

# plt.plot(MR,gMRic,'k')
# plt.plot(LR,gLRic,'r')
# plt.show()



####################################################


# plt.figure(figsize=(18,18))
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

# plt.subplot(221)
# plt.plot(gMRic,wMRic,'ko',label='MR')
# plt.plot(gLRic,wLRic,'rs',label='LR')
# plt.legend()

# plt.subplot(222)
# plt.semilogy(gMRic,np.sum(MR_BRarr,axis=0),'ko',label='MR')
# plt.plot(gLRic,np.sum(LR_BRarr,axis=0),'rs',label='LR')
# plt.legend()

# plt.subplot(223)
# plt.semilogy(gMRic,np.sum(MR_BRarr,axis=0)+np.sum(MR_BRarrsed,axis=0),'ko',label='MR')
# plt.plot(gLRic,np.sum(LR_BRarr,axis=0)+np.sum(LR_BRarrsed,axis=0),'rs',label='LR')
# plt.legend()

# plt.subplot(224)
# plt.plot(gMRic,np.sum(MR_BRarr,axis=0)/(np.sum(MR_BRarr,axis=0)+np.sum(MR_BRarrsed,axis=0)),'ko',label='MR')
# plt.plot(gLRic,np.sum(LR_BRarr,axis=0)/(np.sum(LR_BRarr,axis=0)+np.sum(LR_BRarrsed,axis=0)),'rs',label='LR')
# plt.legend()


# plt.show()





# ########################################
# plt.figure(figsize=(16,20))
# plt.subplots_adjust(wspace=0.2, hspace=0.2)

# plt.subplot(421)
# plt.plot(MR, np.sum(MR_BRarr,axis=0),'k-', label='MR', lw=2)



# plt.figure(figsize=(18,6))
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

# plt.subplot(221)
# rec=Rectangle((11,0), 1, 500, clip_on=False, color='gray')
# plt.gca().add_artist(rec)

# plt.plot(MR, wMRic,'k',label='MR') #-np.max(zMR)
# plt.plot(LR, wLRic,'r--', lw=2, label='LR') 
# plt.title('a) ', loc='left')
# plt.ylabel('Width (m)'); plt.xlabel('River kilometer')
# plt.legend()
# plt.gca().invert_xaxis()
# plt.ylim(0,500)
# plt.text(11,100,'former\nLake\nAldwell')




## river gradient versus wood abundance





## river gradient versus persistence

