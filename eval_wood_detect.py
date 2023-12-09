## Dan Buscombe, Marda Science
## 2023
import os, json, sys
import rioxarray
import xarray as xr 
from glob import glob 
import matplotlib.pyplot as plt
from dask.distributed import Client
from tqdm import tqdm
import numpy as np
import pandas as pd
from area import area 
from skimage.measure import label, regionprops_table


#############################################################
#############################################################
#############################################################
#################### user inputs 

dtype = 'float64'
chunksize = ("auto", "auto")

times = [
    '2012-04-07',
    '2017-09-22'
]

# n_workers = 20
# threads_per_worker = 2
# memory_limit='100GB'

cwd = os.getcwd()

## factor that converts grid uints 1/8 x 1/8
# into units 1 x 1, i.e. 8 x 8
grid2sqm = 64

#############################################################
# if do_analysis:
## start client
# client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

# Create variable used for time axis
time_var = xr.Variable('time',times)

######### get regions and clipper
regions = sorted(glob('../raw_data/GIS/LR*ID*_epsg6339.geojson'))
regions = [r for r in regions if 'pts' not in r]
print("{} regions".format(len(regions)))

geometries = []
for r in regions:
    with open(r) as f:
        gj = json.load(f)
    features = gj['features'][0]

    geometries.append(features['geometry'])


brfile = '../results/LR/LR_wood/wood_detect/model1/LR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
budget_reaches = gj['features']

brfile = '../results/MR/MR_wood/wood_detect/model1/MR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
MRbudget_reaches = gj['features']

gcfile = '../results/LR/LR_wood/wood_detect/model1/LR_global_clipper.geojson'
with open(gcfile) as f:
    gj = json.load(f)
global_clipper = gj['features']

budget_reaches_redo = []
for b in budget_reaches:
    budget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))

MRbudget_reaches_redo = []
for b in MRbudget_reaches:
    MRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))

# get area of each budget reach and  put in a list
A_LR = []
for g in tqdm(budget_reaches):
    A_LR.append(area(g['geometry']))

A_MR = []
for g in tqdm(MRbudget_reaches):
    A_MR.append(area(g['geometry']))

## distances downstrean for each BR
dists = pd.read_csv('br_dists.csv')
LR = np.hstack((0,np.array(dists['LR'])))
MR = np.hstack((0,np.array(dists['MR'][:43])))


### simple comparison between loads

### compute percent errors
def prc_err(o,e):
    return np.abs((o-e)/e)*100


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
### LR


######### observed
LR_gt_20170922 = rioxarray.open_rasterio("../raw_data/dig_wood/LR_20170922_dig_wood_clipped_active_budgetextent.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
LR_gt_20120407 = rioxarray.open_rasterio("../raw_data/dig_wood/LR_20120407_dig_wood_clipped_active_budgetextent.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

### sum of all wood pixels is the target metric
LR_target_gt_20170922 = LR_gt_20170922[1].sum().compute().to_numpy()
LR_target_gt_20120407 = LR_gt_20120407[1].sum().compute().to_numpy()


label_img = label(LR_gt_20120407[1]==1)
props = regionprops_table(label_img, properties=('area','axis_minor_length'))
a = props['area'][np.where(props['area']<100000)[0]]
LR_target_gt_20120407_large = np.sum(a[np.where(a>4096)[0]])

label_img = label(LR_gt_20170922[1]==1)
props = regionprops_table(label_img, properties=('area','axis_minor_length'))
a = props['area'][np.where(props['area']<100000)[0]]
LR_target_gt_20170922_large = np.sum(a[np.where(a>4096)[0]])


######### estimated
LR_20170922 = rioxarray.open_rasterio("../results/LR/LR_wood/wood_detect/model1/LR_20170922_epsg6339_cleaned.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
LR_20120407 = rioxarray.open_rasterio("../results/LR/LR_wood/wood_detect/model1/LR_20120407_epsg6339_cleaned.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

# ### sum of all wood pixels is the target metric
LR_est_gt_20170922 = LR_20170922[1].sum().compute().to_numpy()
LR_est_gt_20120407 = LR_20120407[1].sum().compute().to_numpy()


label_img = label(LR_20120407[1]==1)
props = regionprops_table(label_img, properties=('area','axis_minor_length'))
a = props['area'][np.where(props['area']<100000)[0]]
LR_est_gt_20120407_large = np.sum(a[np.where(a>4096)[0]])

label_img = label(LR_20170922[1]==1)
props = regionprops_table(label_img, properties=('area','axis_minor_length'))
a = props['area'][np.where(props['area']<100000)[0]]
LR_est_gt_20170922_large = np.sum(a[np.where(a>4096)[0]])


LR_20170922_prc_err_all = prc_err(LR_target_gt_20170922, LR_est_gt_20170922)
LR_20170922_prc_err_large = prc_err(LR_target_gt_20170922_large, LR_est_gt_20170922_large)

LR_20120407_prc_err_all = prc_err(LR_target_gt_20120407, LR_est_gt_20120407)
LR_20120407_prc_err_large = prc_err(LR_target_gt_20120407_large, LR_est_gt_20120407_large)


props_obs_LR20170922 = props_df(LR_gt_20170922[1].to_numpy())
props_obs_LR20120407 = props_df(LR_gt_20120407[1].to_numpy())
props_est_LR20170922 = props_df(LR_20170922[1].to_numpy())
props_est_LR20120407 = props_df(LR_20120407[1].to_numpy())

LR_frq1, bins1, ax = plt.hist(props_obs_LR20170922.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)
del ax
LR_frq2, bins, ax = plt.hist(props_est_LR20170922.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)    
del ax

LR_frq3, bins1, ax = plt.hist(props_obs_LR20120407.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)
del ax
LR_frq4, bins, ax = plt.hist(props_est_LR20120407.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)    
del ax

plt.close('all')

print(f"Observed, LR, 17/09/22: {LR_target_gt_20170922}")
print(f"Observed, LR, 12/04/07: {LR_target_gt_20120407}")

print(f"Estimated, LR, 17/09/22: {LR_est_gt_20170922}")
print(f"Estimated, LR, 12/04/07: {LR_est_gt_20120407}")

print(f"Observed large wood, LR, 17/09/22: {LR_target_gt_20170922_large}")
print(f"Observed large wood, LR, 12/04/07: {LR_target_gt_20120407_large}")

print(f"Estimated large wood, LR, 17/09/22: {LR_est_gt_20170922_large}")
print(f"Estimated large wood, LR, 12/04/07: {LR_est_gt_20120407_large}")

print(f"% err. all wood, LR, 17/09/22: {LR_20170922_prc_err_all}")
print(f"% err. all wood, LR, 12/04/07: {LR_20120407_prc_err_all}")

print(f"% err. large wood, LR, 17/09/22: {LR_20170922_prc_err_large}")
print(f"% err. large wood, LR, 12/04/07: {LR_20120407_prc_err_large}")

#############################################################
### MR

######### observed
MR_gt_20170922 = rioxarray.open_rasterio("../raw_data/dig_wood/MR_20170922_dig_wood_clipped_active_budgetextent.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
MR_gt_20120407 = rioxarray.open_rasterio("../raw_data/dig_wood/MR_20120407_dig_wood_clipped_active_budgetextent.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

### sum of all wood pixels is the target metric
MR_target_gt_20170922 = MR_gt_20170922[1].sum().compute().to_numpy()
MR_target_gt_20120407 = MR_gt_20120407[1].sum().compute().to_numpy()


label_img = label(MR_gt_20120407[1]==1)
props = regionprops_table(label_img, properties=('area','axis_minor_length'))
a = props['area'][np.where(props['area']<100000)[0]]
MR_target_gt_20120407_large = np.sum(a[np.where(a>4096)[0]])

label_img = label(MR_gt_20170922[1]==1)
props = regionprops_table(label_img, properties=('area','axis_minor_length'))
a = props['area'][np.where(props['area']<100000)[0]]
MR_target_gt_20170922_large = np.sum(a[np.where(a>4096)[0]])

######### estimated
MR_20170922 = rioxarray.open_rasterio("../results/MR/MR_wood/wood_detect/model1/MR_20170922_epsg6339_cleaned.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
MR_20120407 = rioxarray.open_rasterio("../results/MR/MR_wood/wood_detect/model1/MR_20120407_epsg6339_cleaned.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

# ### sum of all wood pixels is the target metric
MR_est_gt_20170922 = MR_20170922[1].sum().compute().to_numpy()
MR_est_gt_20120407 = MR_20120407[1].sum().compute().to_numpy()

label_img = label(MR_20120407[1]==1)
props = regionprops_table(label_img, properties=('area','axis_minor_length'))
a = props['area'][np.where(props['area']<100000)[0]]
MR_est_gt_20120407_large = np.sum(a[np.where(a>4096)[0]])

label_img = label(MR_20170922[1]==1)
props = regionprops_table(label_img, properties=('area','axis_minor_length'))
a = props['area'][np.where(props['area']<100000)[0]]
MR_est_gt_20170922_large = np.sum(a[np.where(a>4096)[0]])

MR_20170922_prc_err_all = prc_err(MR_target_gt_20170922, MR_est_gt_20170922)
MR_20170922_prc_err_large = prc_err(MR_target_gt_20170922_large, MR_est_gt_20170922_large)

MR_20120407_prc_err_all = prc_err(MR_target_gt_20120407, MR_est_gt_20120407)
MR_20120407_prc_err_large = prc_err(MR_target_gt_20120407_large, MR_est_gt_20120407_large)

print(f"Observed, MR, 17/09/22: {MR_target_gt_20170922}")
print(f"Observed, MR, 12/04/07: {MR_target_gt_20120407}")

print(f"Estimated, MR, 17/09/22: {MR_est_gt_20170922}")
print(f"Estimated, MR, 12/04/07: {MR_est_gt_20120407}")

print(f"Observed large wood, MR, 17/09/22: {MR_target_gt_20170922_large}")
print(f"Observed large wood, MR, 12/04/07: {MR_target_gt_20120407_large}")

print(f"Estimated large wood, MR, 17/09/22: {MR_est_gt_20170922_large}")
print(f"Estimated large wood, MR, 12/04/07: {MR_est_gt_20120407_large}")

print(f"% err. all wood, MR, 17/09/22: {MR_20170922_prc_err_all}")
print(f"% err. all wood, MR, 12/04/07: {MR_20120407_prc_err_all}")

print(f"% err. large wood, MR, 17/09/22: {MR_20170922_prc_err_large}")
print(f"% err. large wood, MR, 12/04/07: {MR_20120407_prc_err_large}")

props_est_MR20170922 = props_df(MR_20170922[1].to_numpy())
props_est_MR20120407 = props_df(MR_20120407[1].to_numpy())
props_obs_MR20170922 = props_df(MR_gt_20170922[1].to_numpy())
props_obs_MR20120407 = props_df(MR_gt_20120407[1].to_numpy())


MR_frq1, bins1, ax = plt.hist(props_obs_MR20170922.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)
del ax
MR_frq2, bins, ax = plt.hist(props_est_MR20170922.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)    
del ax

MR_frq3, bins1, ax = plt.hist(props_obs_MR20120407.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)
del ax
MR_frq4, bins, ax = plt.hist(props_est_MR20120407.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)    
del ax

plt.close('all')

# Observed, LR, 17/09/22: 3214603.0
# Observed, LR, 12/04/07: 3239358.0
# Estimated, LR, 17/09/22: 2837693.0
# Estimated, LR, 12/04/07: 2841303.0
# Observed large wood, LR, 17/09/22: 2194766
# Observed large wood, LR, 12/04/07: 1900046
# Estimated large wood, LR, 17/09/22: 1915133
# Estimated large wood, LR, 12/04/07: 1821309
# % err. all wood, LR, 17/09/22: 13.28226774930954
# % err. all wood, LR, 12/04/07: 14.009593427181244
# % err. large wood, LR, 17/09/22: 14.601231350511949
# % err. large wood, LR, 12/04/07: 4.3230994850407045

# Observed, MR, 17/09/22: 2805392.0
# Observed, MR, 12/04/07: 828287.0
# Estimated, MR, 17/09/22: 2188247.0
# Estimated, MR, 12/04/07: 979531.0
# Observed large wood, MR, 17/09/22: 1529060.0
# Observed large wood, MR, 12/04/07: 501762.0
# Estimated large wood, MR, 17/09/22: 1320984.0
# Estimated large wood, MR, 12/04/07: 645910.0
# % err. all wood, MR, 17/09/22: 28.202712535858154
# % err. all wood, MR, 12/04/07: 15.440450608730316
# % err. large wood, MR, 17/09/22: 15.751591238046789
# % err. large wood, MR, 12/04/07: 22.317041073833817

## mean accuracy
## 15.25

# LR_BR=[]; LR_BR_=[]
# for g in tqdm(budget_reaches_redo):
#     wood_gt = LR_gt_20120407.rio.clip([g], LR_gt_20120407.rio.crs)
#     # result = (wood_gt[1]).sum().compute().to_numpy() 
#     # BR.append(float(result))
#     label_img = label(wood_gt[1]==1)
#     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     a = props['area'][np.where(props['area']<100000)[0]]
#     result_ = np.sum(a[np.where(a>4096)[0]])
#     result = np.sum(a)                
#     LR_BR.append(result)
#     LR_BR_.append(result_)

# LR_BR2=[]; LR_BR2_=[]
# for g in tqdm(budget_reaches_redo):
#     wood_gt = LR_gt_20170922.rio.clip([g], LR_gt_20170922.rio.crs)
#     label_img = label(wood_gt[1]==1)
#     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     a = props['area'][np.where(props['area']<100000)[0]]
#     result_ = np.sum(a[np.where(a>4096)[0]])
#     result = np.sum(a)                
#     LR_BR2.append(result)
#     LR_BR2_.append(result_)

# MR_BR=[]; MR_BR_=[]
# for g in tqdm(MRbudget_reaches_redo):
#     wood_gt = MR_gt_20120407.rio.clip([g], MR_gt_20120407.rio.crs)
#     label_img = label(wood_gt[1]==1)
#     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     a = props['area'][np.where(props['area']<100000)[0]]
#     result_ = np.sum(a[np.where(a>4096)[0]])
#     result = np.sum(a)                
#     MR_BR.append(result)
#     MR_BR_.append(result_)

# MR_BR2=[]; MR_BR2_=[]
# for g in tqdm(MRbudget_reaches_redo):
#     wood_gt = MR_gt_20170922.rio.clip([g], MR_gt_20170922.rio.crs)
#     label_img = label(wood_gt[1]==1)
#     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     a = props['area'][np.where(props['area']<100000)[0]]
#     result_ = np.sum(a[np.where(a>4096)[0]])
#     result = np.sum(a)                
#     MR_BR2.append(result)
#     MR_BR2_.append(result_)

# np.savez('summaries/MR_meas_wood_budget.npz',MR_target_gt_20170922=MR_target_gt_20170922, MR_target_gt_20120407=MR_target_gt_20120407, MR_est_gt_20120407=MR_est_gt_20120407, MR_est_gt_20170922=MR_est_gt_20170922, MR_BR2=MR_BR2, MR_BR=MR_BR) #threshes=threshes, MR_S=MR_S, 
        
# np.savez('summaries/LR_meas_wood_budget.npz', LR_target_gt_20170922=LR_target_gt_20170922, LR_target_gt_20120407=LR_target_gt_20120407, LR_est_gt_20120407=LR_est_gt_20120407, LR_est_gt_20170922=LR_est_gt_20170922, LR_BR2=LR_BR2, LR_BR=LR_BR) #threshes=threshes, S=S, 


with np.load('summaries/MR_meas_wood_budget.npz', allow_pickle=True) as f:
    MR_target_gt_20170922 = f['MR_target_gt_20170922']
    MR_target_gt_20120407 = f['MR_target_gt_20120407']
    MR_est_gt_20120407 = f['MR_est_gt_20120407']
    MR_est_gt_20170922 = f['MR_est_gt_20170922']
    MR_BR2 = f['MR_BR2']
    MR_BR = f['MR_BR']


# LR_estBR=[]; LR_estBR_=[]
# for g in tqdm(budget_reaches_redo):
#     wood_gt = LR_20120407.rio.clip([g], LR_20120407.rio.crs)
#     label_img = label(wood_gt[1]==1)
#     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     a = props['area'][np.where(props['area']<3000000)[0]]
#     LR_estBR_.append(np.sum(a[np.where(a>4096)[0]]))
#     LR_estBR.append(np.sum(a))

# LR_estBR2=[]; LR_estBR2_=[]
# for g in tqdm(budget_reaches_redo):
#     wood_gt = LR_20170922.rio.clip([g], LR_20170922.rio.crs)
#     label_img = label(wood_gt[1]==1)
#     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     a = props['area'][np.where(props['area']<3000000)[0]]
#     LR_estBR2_.append(np.sum(a[np.where(a>4096)[0]]))
#     LR_estBR2.append(np.sum(a) )

# MR_estBR=[]; MR_estBR_=[]
# for g in tqdm(MRbudget_reaches_redo):
#     wood_gt = MR_20120407.rio.clip([g], MR_20120407.rio.crs)
#     label_img = label(wood_gt[1]==1)
#     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     a = props['area'][np.where(props['area']<3000000)[0]]
#     MR_estBR_.append(np.sum(a[np.where(a>4096)[0]]))
#     MR_estBR.append(np.sum(a))

# MR_estBR2=[]; MR_estBR2_=[]
# for g in tqdm(MRbudget_reaches_redo):
#     wood_gt = MR_20170922.rio.clip([g], MR_20170922.rio.crs)
#     label_img = label(wood_gt[1]==1)
#     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     a = props['area'][np.where(props['area']<3000000)[0]]
#     MR_estBR2_.append(np.sum(a[np.where(a>4096)[0]]))
#     MR_estBR2.append(np.sum(a))    


# np.savez('summaries/MR_distribution_area_length_wood_eval.npz', props_gt_MR20120407=props_obs_MR20120407, props_gt_MR20170922=props_obs_MR20170922, props_est_MR20120407=props_est_MR20120407, props_est_MR20170922=props_est_MR20170922)

# np.savez('summaries/LR_distribution_area_length_wood_eval.npz', props_gt_LR20120407=props_obs_LR20120407, props_gt_LR20170922=props_obs_LR20170922, props_est_LR20120407=props_est_LR20120407, props_est_LR20170922=props_est_LR20170922)


# np.savez('summaries/MR_eval_wood_budget.npz', MR_target_gt_20120407 = MR_target_gt_20120407, MR_target_gt_20170922=MR_target_gt_20170922, MR_estBR2=MR_estBR2, MR_estBR=MR_estBR, MRbudget_reaches=MRbudget_reaches_redo, grid2sqm=grid2sqm)

# np.savez('summaries/LR_eval_wood_budget.npz', LR_target_gt_20120407 = LR_target_gt_20120407, LR_target_gt_20170922=LR_target_gt_20170922, LR_estBR2=LR_estBR2, LR_estBR=LR_estBR, LRbudget_reaches=budget_reaches_redo, grid2sqm=grid2sqm)

with np.load('summaries/MR_eval_wood_budget.npz', allow_pickle=True) as f:
    MR_target_gt_20120407 = f['MR_target_gt_20120407']
    MR_target_gt_20170922 = f['MR_target_gt_20170922']
    MR_estBR2 = f['MR_estBR2']
    MR_estBR = f['MR_estBR']


with np.load('summaries/LR_eval_wood_budget.npz', allow_pickle=True) as f:
    LR_target_gt_20120407 = f['LR_target_gt_20120407']
    LR_target_gt_20170922 = f['LR_target_gt_20170922']
    LR_estBR2 = f['LR_estBR2']
    LR_estBR = f['LR_estBR']


with np.load('summaries/LR_meas_wood_budget.npz', allow_pickle=True) as f:
    LR_target_gt_20170922 = f['LR_target_gt_20170922']
    LR_target_gt_20120407 = f['LR_target_gt_20120407']
    LR_est_gt_20120407 = f['LR_est_gt_20120407']
    LR_est_gt_20170922 = f['LR_est_gt_20170922']
    LR_BR2 = f['LR_BR2']
    LR_BR = f['LR_BR']



# O = [MR_target_gt_20120407,MR_target_gt_20170922,LR_target_gt_20120407,LR_target_gt_20170922]
# E = [MR_est_gt_20120407,MR_est_gt_20170922,LR_est_gt_20120407,LR_est_gt_20170922]

ovdf = 1.0 #1.2

E = [np.cumsum(np.array(MR_estBR)/grid2sqm)[-1],np.cumsum(np.array(MR_estBR2)/grid2sqm)[-1],np.cumsum(np.array(LR_estBR)/grid2sqm)[-1],np.cumsum(np.array(LR_estBR2)/grid2sqm)[-1]]
O = [np.cumsum(np.array(MR_BR)/grid2sqm)[-1],np.cumsum(np.array(MR_BR2)/grid2sqm)[-1],np.cumsum(np.array(LR_BR)/grid2sqm)[-1],np.cumsum(np.array(LR_BR2)/grid2sqm)[-1]]

O = np.array(O)/ovdf


###########==================================================
plt.figure(figsize=(14,14))
plt.subplots_adjust(wspace=0.7, hspace=0.2)

ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax1.loglog(O[0],E[0],'ko',label='MR, 2012-04-07')
ax1.plot(O[1],E[1],'ks',label='MR, 2017-09-22')
ax1.plot(O[2],E[2],'ro',label='LR, 2012-04-07')
ax1.plot(O[3],E[3],'rs',label='LR, 2017-09-22')

yl=plt.xlim()
ax1.plot(yl, yl, 'b:', lw=2, label='1:1 relation')

# A = np.vstack([np.array(O), np.ones(len(O))]).T
# m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]

# ax1.plot(np.sort(np.array(O)), m*np.sort(np.array(O)) + c, 'r:',lw=2, label='y = '+str(m)[:4]+'x+'+str(c)[:4])

# ### inverse prob
# A = np.vstack([np.array(E), np.ones(len(E))]).T
# m, c = np.linalg.lstsq(A, np.array(O), rcond=None)[0]

ax1.text(O[0],10+E[0],"14.09%") #14.09
ax1.text(O[1],10+E[1],"13.28%") #13.28
ax1.text(O[2],10+E[2],"14.32%") #14.32
ax1.text(O[3],10+E[3],"14.60%") #14.60

ax1.legend()
ax1.set_ylabel(r"Estimated wood, m$^2$")
ax1.set_xlabel(r"Observed wood, m$^2$")
ax1.set_title('a) MR+LR', loc='left')

ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax2.loglog(bins1[1:]/grid2sqm, MR_frq3,'m-',lw=2, label='Obs., 2012-04-07')
ax2.plot(bins1[1:]/grid2sqm, MR_frq4,'b-',lw=2, label='Est., 2012-04-07')

ax2.plot(bins1[1:]/grid2sqm, MR_frq1,'m--',lw=2, label='Obs., 2017-09-22')
ax2.plot(bins1[1:]/grid2sqm, MR_frq2,'b--',lw=2, label='Est., 2017-09-22')
ax2.legend()
ax2.set_ylabel(r'Frequency')
ax2.set_xlabel(r"Wood pile or piece area (m$^2$)")
ax2.set_title('b) MR', loc='left')

ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax3.loglog(bins1[1:]/grid2sqm, LR_frq3,'m-',lw=2, label='Obs. 2012-04-07')
ax3.plot(bins1[1:]/grid2sqm, LR_frq4,'b-',lw=2, label='Est. 2012-04-07')

ax3.plot(bins1[1:]/grid2sqm, LR_frq1,'m--',lw=2, label='Obs. 2017-09-22')
ax3.plot(bins1[1:]/grid2sqm, LR_frq2,'b--',lw=2, label='Est. 2017-09-22')
ax3.legend()
ax3.set_ylabel(r'Frequency')
ax3.set_xlabel(r"Wood pile or piece area (m$^2$)")
ax3.set_title('c) LR', loc='left')

ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax4.plot(MR,np.cumsum(np.array(MR_BR)/grid2sqm), 'm-', label='Obs.'+times[0])
ax4.plot(MR,np.cumsum(np.array(MR_estBR)/grid2sqm), 'm--', label='Est. '+times[0])

ax4.plot(MR,np.cumsum(np.array(MR_BR2)/grid2sqm), 'b-', lw=2, label='Obs. '+times[1])
ax4.plot(MR, np.cumsum(np.array(MR_estBR2)/grid2sqm), 'b--', lw=2, label='Est. '+times[1])
ax4.set_title('d) MR', loc='left')
plt.legend()
ax4.set_xlabel("Distance downstream (km)")
ax4.set_ylabel(r"Cumulative sum of wood, m$^2$")
plt.ylim(0,40000)
plt.xlim(0,8)

ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
ax5.plot(LR,np.cumsum(np.array(LR_BR)/grid2sqm), 'm-', label='Obs. '+times[0])
ax5.plot(LR,np.cumsum(np.array(LR_estBR)/grid2sqm), 'm--', label='Est. '+times[0])

ax5.plot(LR,np.cumsum(np.array(LR_BR2)/grid2sqm), 'b-', lw=2, label='Obs. '+times[1])
ax5.plot(LR,np.cumsum(np.array(LR_estBR2)/grid2sqm), 'b--', lw=2, label='Est. '+times[1])
ax5.set_title('e) LR', loc='left')
ax5.set_xlabel("Distance downstream (km)")
ax5.set_ylabel(r"Cumulative sum of wood, m$^2$")
plt.legend()
plt.ylim(0,40000)
plt.xlim(0,8)

# plt.show()
plt.savefig("summaries/MR_LR_wood_eval.png", dpi=300, bbox_inches="tight")
plt.close()




# #############################################################
# #############################################################

#############################################################
### LR
# get filtered wood probs, clipped to margins
wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/model2/Elwha_LR_*wood_filtered_prob_regrid.tif'))
print(len(wood_files))

wood_files = [w for w in wood_files for t in times if t in w]

print(wood_files)

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')

# Rename the variable to a more useful name
wood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

# #############################################################
### MR
# get filtered wood probs, clipped to margins
wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/model2/Elwha_MR_*wood_filtered_prob_regrid.tif'))
print(len(wood_files))

wood_files = [w for w in wood_files for t in times if t in w]
print(wood_files)

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')

# Rename the variable to a more useful name
MRwood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

#############################################################
#############################################################


## any blob with area > 100000 is considered error
## blobs > 4096 area (< 64x64 px) are accounted for separately

# threshes = np.arange(.05,.6,.05)
# S=[]; S_ = []
# for thres in threshes:
#     for time in times:

#         image = wood_geotiffs_ds.wood.sel(time=time).to_numpy()
#         label_img = label(image>thres)
#         props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#         a = props['area'][np.where(props['area']<100000)[0]]
#         result_ = np.sum(a[np.where(a>4096)[0]])
#         result = np.sum(a)                
#         # result = (image>thres).sum()#.compute().to_numpy()
#         print(f"{thres}: {result}")
#         S.append(float(result))
#         S_.append(float(result_))
#         del image, label_img, a, result, result_, props


# MR_S=[]; MR_S_ = []
# for thres in threshes:
#     for time in times:

#         image = MRwood_geotiffs_ds.wood.sel(time=time).to_numpy()
#         label_img = label(image>thres)
#         props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#         a = props['area'][np.where(props['area']<100000)[0]]
#         result_ = np.sum(a[np.where(a>4096)[0]])
#         result = np.sum(a)                
#         # result = (image>thres).sum()#.compute().to_numpy()
#         print(f"{thres}: {result}")
#         MR_S.append(float(result))
#         MR_S_.append(float(result_))
#         del image, label_img, a, result, result_, props


# np.savez('summaries/MR_eval_wood_prob_thres.npz', MR_S = MR_S, MR_S_=MR_S_, threshes=threshes)

with np.load('summaries/MR_eval_wood_prob_thres.npz', allow_pickle=True) as f:
    MR_S = f['MR_S']
    MR_S_ = f['MR_S_']
    threshes = f['threshes']

np.savez('summaries/LR_eval_wood_prob_thres.npz', LR_S = S, LR_S_=S_, threshes=threshes)

with np.load('summaries/LR_eval_wood_prob_thres.npz', allow_pickle=True) as f:
    LR_S = f['LR_S']
    LR_S_ = f['LR_S_']
    threshes = f['threshes']


f1=2.2
plt.figure(figsize=(8,8))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(221)
plt.semilogy(threshes, (np.array(LR_S[::2])/grid2sqm)/f1, 'r-', lw=1, label='Est, LR, all, '+times[0])
plt.plot(threshes, (np.array(LR_S_[::2])/grid2sqm)/f1, 'b:', lw=1, label='Est, LR, large, '+times[0])

plt.axhline(y=O[0], color='r', linestyle='--', label='Obs, LR, '+times[0])
plt.axvline(x=.25, linestyle='--')
plt.xlabel('Threshold wood probability')
plt.ylabel(r"Wood, m$^2$"); #plt.ylim(7500,35000)
plt.legend()
plt.title('a)', loc='left')

plt.subplot(222)
plt.semilogy(threshes, (np.array(LR_S[1::2])/grid2sqm)/f1, 'k-', lw=2, label='Est, LR, all, '+times[1])
plt.plot(threshes, (np.array(LR_S_[1::2])/grid2sqm)/f1, 'b:', lw=2, label='Est, LR, large, '+times[1])
plt.axhline(y=O[1], color='k', linestyle='--', label='Obs, LR, '+times[1])
plt.axvline(x=.25, linestyle='--')
plt.xlabel('Threshold wood probability')
plt.ylabel(r"Wood, m$^2$"); #plt.ylim(7500,35000)
plt.legend()
plt.title('b)', loc='left')

plt.subplot(223)
plt.semilogy(threshes, (np.array(MR_S[::2])/grid2sqm), 'r-', lw=1, label='Est, MR, all, '+times[0])
plt.plot(threshes, (np.array(MR_S_[::2])/grid2sqm), 'b:', lw=1, label='Est, MR, large, '+times[0])

plt.axhline(y=O[2], color='r', linestyle='--', label='Obs, LR, '+times[0])
plt.axvline(x=.25, linestyle='--')
plt.xlabel('Threshold wood probability')
plt.ylabel(r"Wood, m$^2$"); #plt.ylim(7500,35000)
plt.legend()
plt.title('c)', loc='left')

plt.subplot(224)
plt.semilogy(threshes, (np.array(MR_S[1::2])/grid2sqm)/f1, 'k-', lw=2, label='Est, MR, all, '+times[1])
plt.plot(threshes, (np.array(MR_S_[1::2])/grid2sqm)/f1, 'b:', lw=2, label='Est, MR, large, '+times[1])
plt.axhline(y=O[3], color='k', linestyle='--', label='Obs, MR, '+times[1])
plt.axvline(x=.25, linestyle='--')
plt.xlabel('Threshold wood probability')
plt.ylabel(r"Wood, m$^2$"); #plt.ylim(7500,35000)
plt.legend()
plt.title('d)', loc='left')

# plt.show()
plt.savefig("summaries/MR_LR_wood_eval_thres_prob.png", dpi=300, bbox_inches="tight")
plt.close()


    # LR_20170922 = rioxarray.open_rasterio("../results/LR/LR_wood/wood_detect/Elwha_LR_2017-09-22_wood_filtered_bin0.1_regrid_final.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
    # LR_20120407 = rioxarray.open_rasterio("../results/LR/LR_wood/wood_detect/Elwha_LR_2012-04-07_wood_filtered_bin0.1_regrid_final.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

    # LR_20170922 = rioxarray.open_rasterio("../results/LR/LR_wood/wood_detect/Elwha_LR_2017-09-22_wood_filtered_bin0.15_regrid_final.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
    # LR_20120407 = rioxarray.open_rasterio("../results/LR/LR_wood/wood_detect/Elwha_LR_2012-04-07_wood_filtered_bin0.15_regrid_final.tif", chunks=chunksize, dtype=dtype).to_dataset('band')


#     #############################################################
#     ### MR

#     # MRgt_20170922 = rioxarray.open_rasterio("../raw_data/dig_wood/MR_20170922_dig_wood_clipped_active_budgetextent.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
#     # MRgt_20120407 = rioxarray.open_rasterio("../raw_data/dig_wood/MR_20120407_dig_wood_clipped_active_budgetextent_v2.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

#     # # ### sum of all wood pixels is the target metric
#     # # MRtarget_gt_20170922 = MRgt_20170922[1].sum().compute().to_numpy()
#     # # MRtarget_gt_20120407 = MRgt_20120407[1].sum().compute().to_numpy()

#     # label_img = label(MRgt_20120407[1]==1)
#     # props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     # a = props['area'][np.where(props['area']<100000)[0]]
#     # MRtarget_gt_20120407_ = np.sum(a[np.where(a>4096)[0]])
#     # MRtarget_gt_20120407 = np.sum(a)                

#     # label_img = label(MRgt_20170922[1]==1)
#     # props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     # a = props['area'][np.where(props['area']<100000)[0]]
#     # MRtarget_gt_20170922_ = np.sum(a[np.where(a>4096)[0]])
#     # MRtarget_gt_20170922 = np.sum(a)    

#     threshes = np.arange(.05,.5,.01)
#     MR_S=[]; MR_S_ = []
#     for thres in threshes:
#         for time in times:

#             image = wood_geotiffs_ds.wood.sel(time=time).to_numpy()
#             label_img = label(image>thres)
#             props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#             a = props['area'][np.where(props['area']<100000)[0]]
#             result_ = np.sum(a[np.where(a>4096)[0]])
#             result = np.sum(a)                
#             # result = (image>thres).sum()#.compute().to_numpy()
#             print(f"{thres}: {result}")
#             MR_S.append(float(result))
#             MR_S_.append(float(result_))

#     # MR_20170922 = rioxarray.open_rasterio("../results/MR/MR_wood/wood_detect/Elwha_MR_2017-09-22_wood_filtered_bin0.1_regrid_final.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
#     # MR_20120407 = rioxarray.open_rasterio("../results/MR/MR_wood/wood_detect/Elwha_MR_2012-04-07_wood_filtered_bin0.1_regrid_final.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

#     # # ### sum of all wood pixels is the target metric
#     # # MRest_gt_20170922 = MR_20170922[1].sum().compute().to_numpy()
#     # # MRest_gt_20120407 = MR_20120407[1].sum().compute().to_numpy()

#     # label_img = label(MR_20120407[1]==1)
#     # props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     # a = props['area'][np.where(props['area']<100000)[0]]
#     # MRest_gt_20120407_ = np.sum(a[np.where(a>4096)[0]])
#     # MRest_gt_20120407 = np.sum(a)                

#     # label_img = label(MR_20170922[1]==1)
#     # props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     # a = props['area'][np.where(props['area']<100000)[0]]
#     # MRest_gt_20170922_ = np.sum(a[np.where(a>4096)[0]])
#     # MRest_gt_20170922 = np.sum(a)       

#     #############################################################
#     #############################################################


#     # BR=[]; BR_=[]
#     # for g in tqdm(budget_reaches_redo):
#     #     wood_gt = gt_20120407.rio.clip([g], gt_20120407.rio.crs)
#     #     # result = (wood_gt[1]).sum().compute().to_numpy() 
#     #     # BR.append(float(result))
#     #     label_img = label(wood_gt[1]==1)
#     #     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     #     a = props['area'][np.where(props['area']<100000)[0]]
#     #     result_ = np.sum(a[np.where(a>4096)[0]])
#     #     result = np.sum(a)                
#     #     BR.append(result)
#     #     BR_.append(result_)

#     # BR2=[]; BR2_=[]
#     # for g in tqdm(budget_reaches_redo):
#     #     wood_gt = gt_20170922.rio.clip([g], gt_20170922.rio.crs)
#     #     # result = (wood_gt[1]).sum().compute().to_numpy() 
#     #     # BR2.append(float(result))
#     #     label_img = label(wood_gt[1]==1)
#     #     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     #     a = props['area'][np.where(props['area']<100000)[0]]
#     #     result_ = np.sum(a[np.where(a>4096)[0]])
#     #     result = np.sum(a)                
#     #     BR2.append(result)
#     #     BR2_.append(result_)

#     # MR_BR=[]; MR_BR_=[]
#     # for g in tqdm(MRbudget_reaches_redo):
#     #     wood_gt = MRgt_20120407.rio.clip([g], MRgt_20120407.rio.crs)
#     #     # result = (wood_gt[1]).sum().compute().to_numpy() 
#     #     # MR_BR.append(float(result))
#     #     label_img = label(wood_gt[1]==1)
#     #     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     #     a = props['area'][np.where(props['area']<100000)[0]]
#     #     result_ = np.sum(a[np.where(a>4096)[0]])
#     #     result = np.sum(a)                
#     #     MR_BR.append(result)
#     #     MR_BR_.append(result_)

#     # MR_BR2=[]; MR_BR2_=[]
#     # for g in tqdm(MRbudget_reaches_redo):
#     #     wood_gt = MRgt_20170922.rio.clip([g], MRgt_20170922.rio.crs)
#     #     # result = (wood_gt[1]).sum().compute().to_numpy() 
#     #     # MR_BR2.append(float(result))
#     #     label_img = label(wood_gt[1]==1)
#     #     props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#     #     a = props['area'][np.where(props['area']<100000)[0]]
#     #     result_ = np.sum(a[np.where(a>4096)[0]])
#     #     result = np.sum(a)                
#     #     MR_BR2.append(result)
#     #     MR_BR2_.append(result_)

#     # np.savez('summaries/MR_meas_wood_budget.npz',MRtarget_gt_20170922=MRtarget_gt_20170922, MRtarget_gt_20120407=MRtarget_gt_20120407, MRest_gt_20120407=MRest_gt_20120407, MRest_gt_20170922=MRest_gt_20170922, threshes=threshes, MR_S=MR_S, MR_BR2=MR_BR2, MR_BR=MR_BR)
            
#     # np.savez('summaries/LR_meas_wood_budget.npz', target_gt_20170922=target_gt_20170922, target_gt_20120407=target_gt_20120407, est_gt_20120407=est_gt_20120407, est_gt_20170922=est_gt_20170922, threshes=threshes, S=S, BR2=BR2, BR=BR)


#     # np.savez('summaries/MR_meas_wood_budget_largepiles.npz',MRtarget_gt_20170922_=MRtarget_gt_20170922_, MRtarget_gt_20120407_=MRtarget_gt_20120407_, MRest_gt_20120407_=MRest_gt_20120407_, MRest_gt_20170922_=MRest_gt_20170922_, threshes=threshes, MR_S_=MR_S_, MR_BR2_=MR_BR2_, MR_BR_=MR_BR_)
            
#     # np.savez('summaries/LR_meas_wood_budget_largepiles.npz', target_gt_20170922_=target_gt_20170922_, target_gt_20120407_=target_gt_20120407_, est_gt_20120407_=est_gt_20120407_, est_gt_20170922_=est_gt_20170922_, threshes=threshes, S_=S_, BR2_=BR2_, BR_=BR_)


#     ###############################################################################################
#     ###############################################################################################


#     frq1, bins1, ax = plt.hist(props_gt_MR20120407.area.values/overdig_factor, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)
#     del ax
#     frq2, bins, ax = plt.hist(props_gt_MR20170922.area.values/overdig_factor, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)    
#     del ax
#     frq3, bins2, ax = plt.hist(props_gt_MR20120407.axis_major_length.values/overdig_factor, bins=np.linspace(1,1200,50))#, cumulative=True)#,log=True)
#     del ax
#     frq4, bins, ax = plt.hist(props_gt_MR20170922.axis_major_length.values/overdig_factor, bins=np.linspace(1,1200,50))#, cumulative=True)#,log=True)  
#     del ax
#     frq5, bins3, ax = plt.hist(props_gt_MR20120407.axis_minor_length.values/overdig_factor, bins=np.linspace(1,800,50))#, cumulative=True)#,log=True)
#     del ax
#     frq6, bins, ax = plt.hist(props_gt_MR20170922.axis_minor_length.values/overdig_factor, bins=np.linspace(1,800,50))#, cumulative=True)#,log=True)  
#     del ax
#     plt.close('all')

#     estfrq1, bins1, ax = plt.hist(props_est_MR20120407.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)
#     del ax
#     estfrq2, bins, ax = plt.hist(props_est_MR20170922.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)    
#     del ax
#     estfrq3, bins2, ax = plt.hist(props_est_MR20120407.axis_major_length.values, bins=np.linspace(1,1200,50))#, cumulative=True)#,log=True)
#     del ax
#     estfrq4, bins, ax = plt.hist(props_est_MR20170922.axis_major_length.values, bins=np.linspace(1,1200,50))#, cumulative=True)#,log=True)  
#     del ax
#     estfrq5, bins3, ax = plt.hist(props_est_MR20120407.axis_minor_length.values, bins=np.linspace(1,800,50))#, cumulative=True)#,log=True)
#     del ax
#     estfrq6, bins, ax = plt.hist(props_est_MR20170922.axis_minor_length.values, bins=np.linspace(1,800,50))#, cumulative=True)#,log=True)  
#     del ax
#     plt.close('all')

#     np.savez('summaries/MR_distribution_area_length_wood_eval.npz', props_gt_MR20120407=props_gt_MR20120407, props_gt_MR20170922=props_gt_MR20170922, props_est_MR20120407=props_est_MR20120407, props_est_MR20170922=props_est_MR20170922)

#     ###############################################################################################

#     lrfrq1, bins1, ax = plt.hist(props_gt_LR20120407.area.values/overdig_factor, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)
#     del ax
#     lrfrq2, bins, ax = plt.hist(props_gt_LR20170922.area.values/overdig_factor, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)    
#     del ax
#     lrfrq3, bins2, ax = plt.hist(props_gt_LR20120407.axis_major_length.values/overdig_factor, bins=np.linspace(1,1200,50))#, cumulative=True)#,log=True)
#     del ax
#     lrfrq4, bins, ax = plt.hist(props_gt_LR20170922.axis_major_length.values/overdig_factor, bins=np.linspace(1,1200,50))#, cumulative=True)#,log=True)  
#     del ax
#     lrfrq5, bins3, ax = plt.hist(props_gt_LR20120407.axis_minor_length.values/overdig_factor, bins=np.linspace(1,800,50))#, cumulative=True)#,log=True)
#     del ax
#     lrfrq6, bins, ax = plt.hist(props_gt_LR20170922.axis_minor_length.values/overdig_factor, bins=np.linspace(1,800,50))#, cumulative=True)#,log=True)  
#     del ax
#     plt.close('all')

#     lrestfrq1, bins1, ax = plt.hist(props_est_LR20120407.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)
#     del ax
#     lrestfrq2, bins, ax = plt.hist(props_est_LR20170922.area.values, bins=np.linspace(1,140000,50))#, cumulative=True)#,log=True)    
#     del ax
#     lrestfrq3, bins2, ax = plt.hist(props_est_LR20120407.axis_major_length.values, bins=np.linspace(1,1200,50))#, cumulative=True)#,log=True)
#     del ax
#     lrestfrq4, bins, ax = plt.hist(props_est_LR20170922.axis_major_length.values, bins=np.linspace(1,1200,50))#, cumulative=True)#,log=True)  
#     del ax
#     lrestfrq5, bins3, ax = plt.hist(props_est_LR20120407.axis_minor_length.values, bins=np.linspace(1,800,50))#, cumulative=True)#,log=True)
#     del ax
#     lrestfrq6, bins, ax = plt.hist(props_est_LR20170922.axis_minor_length.values, bins=np.linspace(1,800,50))#, cumulative=True)#,log=True)  
#     del ax
#     plt.close('all')

#     ###############################################################################################

#     np.savez('summaries/LR_distribution_area_length_wood_eval.npz', props_gt_LR20120407=props_gt_LR20120407, props_gt_LR20170922=props_gt_LR20170922, props_est_LR20120407=props_est_LR20120407, props_est_LR20170922=props_est_LR20170922)

#     ###############################################################################################


    # plt.figure(figsize=(12,16))
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # plt.subplot(321)
    # plt.loglog(bins1[1:]/grid2sqm, frq1,'k-',lw=2, label='MR, Obs., 2012-04-07')
    # plt.plot(bins1[1:]/grid2sqm, estfrq1,'k--',lw=2, label='MR, Est., 2012-04-07')
    # plt.plot(bins1[1:]/grid2sqm, lrfrq1,'r-',lw=2, label='LR, Obs., 2012-04-07')
    # plt.plot(bins1[1:]/grid2sqm, lrestfrq1,'r--',lw=2, label='LR, Est., 2012-04-07')
    # plt.legend()
    # plt.ylabel(r'Frequency')
    # plt.xlabel(r"Wood pile or piece area (m$^2$)")
    # plt.title('a)', loc='left')
    # plt.axvline(x=8**2, linestyle=':', color='b', lw=2)

    # plt.subplot(322)
    # plt.loglog(bins1[1:]/grid2sqm, frq2,'k-',lw=2, label='MR, Obs., 2017-09-22')
    # plt.plot(bins1[1:]/grid2sqm, estfrq2,'k--',lw=2, label='MR, Est., 2017-09-22')
    # plt.plot(bins1[1:]/grid2sqm, lrfrq2,'r-',lw=2, label='LR, Obs., 2017-09-22')
    # plt.plot(bins1[1:]/grid2sqm, lrestfrq2,'r--',lw=2, label='LR, Est., 2017-09-22')
    # plt.legend()
    # plt.ylabel(r'Frequency')
    # plt.xlabel(r"Wood pile or piece area (m$^2$)")
    # plt.title('b)', loc='left')
    # plt.axvline(x=8**2, linestyle=':', color='b', lw=2)

    # plt.subplot(323)
    # plt.loglog(bins2[1:]/np.sqrt(grid2sqm), frq3,'k-',lw=2, label='MR, Observed')
    # plt.plot(bins2[1:]/np.sqrt(grid2sqm), estfrq3,'k--',lw=2, label='MR, Estimated')
    # plt.plot(bins2[1:]/np.sqrt(grid2sqm), lrfrq3,'r-',lw=2, label='LR, Observed')
    # plt.plot(bins2[1:]/np.sqrt(grid2sqm), lrestfrq3,'r--',lw=2, label='LR, Estimated')

    # # plt.legend()
    # plt.ylabel(r'Frequency')
    # plt.xlabel(r"Wood pile or piece major length (m)")
    # plt.title('c)', loc='left')
    # plt.axvline(x=8, linestyle=':', color='b', lw=2)

    # plt.subplot(324)
    # plt.loglog(bins2[1:]/np.sqrt(grid2sqm), frq4,'k-',lw=2, label='MR, Observed')
    # plt.plot(bins2[1:]/np.sqrt(grid2sqm), estfrq4,'k--',lw=2, label='MR, Estimated')
    # plt.plot(bins2[1:]/np.sqrt(grid2sqm), lrfrq4,'r-',lw=2, label='LR, Observed')
    # plt.plot(bins2[1:]/np.sqrt(grid2sqm), lrestfrq4,'r--',lw=2, label='LR, Estimated')

    # plt.title('f)', loc='left')
    # # plt.legend()
    # plt.ylabel(r'Frequency')
    # plt.xlabel(r"Wood pile or piece major length (m)")
    # plt.title('d)', loc='left')
    # plt.axvline(x=8, linestyle=':', color='b', lw=2)

    # plt.subplot(325)
    # plt.loglog(bins3[1:]/np.sqrt(grid2sqm), frq5,'k-',lw=2, label='MR, Observed')
    # plt.plot(bins3[1:]/np.sqrt(grid2sqm), estfrq5,'k--',lw=2, label='MR, Estimated')
    # plt.plot(bins3[1:]/np.sqrt(grid2sqm), lrfrq5,'r-',lw=2, label='LR, Observed')
    # plt.plot(bins3[1:]/np.sqrt(grid2sqm), lrestfrq5,'r--',lw=2, label='LR, Estimated')

    # # plt.legend()
    # plt.ylabel(r'Frequency')
    # plt.xlabel(r"Wood pile or piece minor length (m)")
    # plt.title('e)', loc='left')
    # plt.axvline(x=8, linestyle=':', color='b', lw=2)

    # plt.subplot(326)
    # plt.loglog(bins3[1:]/np.sqrt(grid2sqm), frq6,'k-',lw=2, label='MR, Observed')
    # plt.plot(bins3[1:]/np.sqrt(grid2sqm), estfrq6,'k--',lw=2, label='MR, Estimated')
    # plt.plot(bins3[1:]/np.sqrt(grid2sqm), lrfrq6,'r-',lw=2, label='LR, Observed')
    # plt.plot(bins3[1:]/np.sqrt(grid2sqm), lrestfrq6,'r--',lw=2, label='LR, Estimated')

    # # plt.legend()
    # plt.ylabel(r'Frequency')
    # plt.xlabel(r"Wood pile or piece minor length (m)")
    # plt.title('f)', loc='left')
    # plt.axvline(x=8, linestyle=':', color='b', lw=2)

    # # plt.show()

    # plt.savefig("summaries/MR_LR_distribution_area_length_wood_eval.png", dpi=300, bbox_inches="tight")
    # plt.close()

# else:

#     with np.load('summaries/MR_meas_wood_budget.npz', allow_pickle=True) as f:
#         MR_20120407 = f['MR_20120407']
#         MR_20170922 = f['MR_20170922']
#         threshes = f['threshes']
#         MR_S = f['MR_S']
#         MR_BR2 = f['MR_BR2']
#         MR_BR = f['MR_BR']

#     with np.load('summaries/LR_meas_wood_budget.npz', allow_pickle=True) as f:
#         LR_20120407 = f['LR_20120407']
#         LR_20170922 = f['LR_20170922']
#         # threshes = f['threshes']
#         S = f['S']
#         BR2 = f['BR2']
#         BR = f['BR']



# #################

# if do_analysis:

#     estBR=[]; estBR_=[]
#     for g in tqdm(budget_reaches_redo):
#         wood_gt = LR_20120407.rio.clip([g], LR_20120407.rio.crs)
#         # result = (wood_gt[1]).sum().compute().to_numpy() 
#         # estBR.append(float(result))
#         label_img = label(wood_gt[1]==1)
#         props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#         a = props['area'][np.where(props['area']<3000000)[0]]
#         estBR_.append(np.sum(a[np.where(a>4096)[0]]))
#         estBR.append(np.sum(a))

#     estBR2=[]; estBR2_=[]
#     for g in tqdm(budget_reaches_redo):
#         wood_gt = LR_20170922.rio.clip([g], LR_20170922.rio.crs)
#         # result = (wood_gt[1]).sum().compute().to_numpy() 
#         # estBR2.append(float(result))
#         label_img = label(wood_gt[1]==1)
#         props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#         a = props['area'][np.where(props['area']<3000000)[0]]
#         estBR2_.append(np.sum(a[np.where(a>4096)[0]]))
#         estBR2.append(np.sum(a) )

#     estMR_BR=[]; estMR_BR_=[]
#     for g in tqdm(MRbudget_reaches_redo):
#         wood_gt = MR_20120407.rio.clip([g], MR_20120407.rio.crs)
#         # result = (wood_gt[1]).sum().compute().to_numpy() 
#         # estMR_BR.append(float(result))
#         label_img = label(wood_gt[1]==1)
#         props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#         a = props['area'][np.where(props['area']<3000000)[0]]
#         estMR_BR_.append(np.sum(a[np.where(a>4096)[0]]))
#         estMR_BR.append(np.sum(a))

#     estMR_BR2=[]; estMR_BR2_=[]
#     for g in tqdm(MRbudget_reaches_redo):
#         wood_gt = MR_20170922.rio.clip([g], MR_20170922.rio.crs)
#         # result = (wood_gt[1]).sum().compute().to_numpy() 
#         # estMR_BR2.append(float(result))
#         label_img = label(wood_gt[1]==1)
#         props = regionprops_table(label_img, properties=('area','axis_minor_length'))
#         a = props['area'][np.where(props['area']<3000000)[0]]
#         estMR_BR2_.append(np.sum(a[np.where(a>4096)[0]]))
#         estMR_BR2.append(np.sum(a))    


#     MRtarget_gt_20120407 = np.cumsum(np.array(MR_BR)/grid2sqm)[-1]
#     MRtarget_gt_20170922 = np.cumsum(np.array(MR_BR2)/grid2sqm)[-1]

#     target_gt_20120407 = np.cumsum(np.array(BR)/grid2sqm)[-1]
#     target_gt_20170922 = np.cumsum(np.array(BR2)/grid2sqm)[-1]

#     MRest_gt_20120407 = np.cumsum(np.array(estMR_BR)/grid2sqm)[-1]
#     MRest_gt_20170922 = np.cumsum(np.array(estMR_BR2)/grid2sqm)[-1]

#     est_gt_20120407 = np.cumsum(np.array(estBR)/grid2sqm)[-1]
#     est_gt_20170922 = np.cumsum(np.array(estBR2)/grid2sqm)[-1]

#     #####
#     MRtarget_gt_20120407_ = np.cumsum(np.array(MR_BR_)/grid2sqm)[-1]
#     MRtarget_gt_20170922_ = np.cumsum(np.array(MR_BR2_)/grid2sqm)[-1]

#     target_gt_20120407_ = np.cumsum(np.array(BR_)/grid2sqm)[-1]
#     target_gt_20170922_ = np.cumsum(np.array(BR2_)/grid2sqm)[-1]

#     MRest_gt_20120407_ = np.cumsum(np.array(estMR_BR_)/grid2sqm)[-1]
#     MRest_gt_20170922_ = np.cumsum(np.array(estMR_BR2_)/grid2sqm)[-1]

#     est_gt_20120407_ = np.cumsum(np.array(estBR_)/grid2sqm)[-1]
#     est_gt_20170922_ = np.cumsum(np.array(estBR2_)/grid2sqm)[-1]   


#     print(f"Obs: 2012-04-07: {MRtarget_gt_20120407/overdig_factor}")
#     print(f"Est: 2012-04-07: {MRest_gt_20120407}")

#     print(f"Obs: 2017-09-22: {MRtarget_gt_20170922/overdig_factor}")
#     print(f"Est: 2017-09-22: {MRest_gt_20170922}")

#     print(100*((MRtarget_gt_20120407/overdig_factor)-MRest_gt_20120407)/MRest_gt_20120407)
#     print(100*((MRtarget_gt_20170922/overdig_factor)-MRest_gt_20170922)/MRest_gt_20170922)

#     print(f"Obs: 2012-04-07: {target_gt_20120407/overdig_factor}")
#     print(f"Est: 2012-04-07: {est_gt_20120407}")

#     print(f"Obs: 2017-09-22: {target_gt_20170922/overdig_factor}")
#     print(f"Est: 2017-09-22: {est_gt_20170922}")

#     print(100*((target_gt_20120407/overdig_factor)-est_gt_20120407)/est_gt_20120407)
#     print(100*((target_gt_20170922/overdig_factor)-est_gt_20170922)/est_gt_20170922)


#     #####
#     print(f"Obs: 2012-04-07: {MRtarget_gt_20120407_/overdig_factor}")
#     print(f"Est: 2012-04-07: {MRest_gt_20120407_}")

#     print(f"Obs: 2017-09-22: {MRtarget_gt_20170922_/overdig_factor}")
#     print(f"Est: 2017-09-22: {MRest_gt_20170922_}")

#     print(100*((MRtarget_gt_20120407_/overdig_factor)-MRest_gt_20120407_)/MRest_gt_20120407_)
#     print(100*((MRtarget_gt_20170922_/overdig_factor)-MRest_gt_20170922_)/MRest_gt_20170922_)

#     print(f"Obs: 2012-04-07: {target_gt_20120407_/overdig_factor}")
#     print(f"Est: 2012-04-07: {est_gt_20120407_}")

#     print(f"Obs: 2017-09-22: {target_gt_20170922_/overdig_factor}")
#     print(f"Est: 2017-09-22: {est_gt_20170922_}")

#     print(100*((target_gt_20120407_/overdig_factor)-est_gt_20120407_)/est_gt_20120407_)
#     print(100*((target_gt_20170922_/overdig_factor)-est_gt_20170922_)/est_gt_20170922_)


#     ###############################################################################################

#     np.savez('summaries/MR_eval_wood_budget.npz', MRtarget_gt_20120407 = target_gt_20120407, MRtarget_gt_20170922=target_gt_20170922, estMR_BR2=estMR_BR2, estMR_BR=estMR_BR, MRbudget_reaches=MRbudget_reaches_redo, grid2sqm=grid2sqm, overdig_factor=overdig_factor)

#     np.savez('summaries/LR_eval_wood_budget.npz', LRtarget_gt_20120407 = target_gt_20120407, LRtarget_gt_20170922=target_gt_20170922, estLR_BR2=estBR2, estLR_BR=estBR, LRbudget_reaches=budget_reaches_redo, grid2sqm=grid2sqm, overdig_factor=overdig_factor)


#     np.savez('summaries/MR_eval_wood_budget_largepiles.npz', MRtarget_gt_20120407_ = target_gt_20120407_, MRtarget_gt_20170922_=target_gt_20170922_, estMR_BR2_=estMR_BR2_, estMR_BR_=estMR_BR_, MRbudget_reaches=MRbudget_reaches_redo, grid2sqm=grid2sqm, overdig_factor=overdig_factor)

#     np.savez('summaries/LR_eval_wood_budget_largepiles.npz', LRtarget_gt_20120407_ = target_gt_20120407_, LRtarget_gt_20170922_=target_gt_20170922_, estLR_BR2_=estBR2_, estLR_BR_=estBR_, LRbudget_reaches=budget_reaches_redo, grid2sqm=grid2sqm, overdig_factor=overdig_factor)


# else:
        
#     with np.load('summaries/LR_eval_wood_budget.npz', allow_pickle=True) as f:
#         LRtarget_gt_20120407 = f['LRtarget_gt_20120407']
#         LRtarget_gt_20170922 = f['LRtarget_gt_20170922']
#         estBR2 = f['estLR_BR2']
#         estBR = f['estLR_BR']


#     with np.load('summaries/MR_eval_wood_budget.npz', allow_pickle=True) as f:
#         MRtarget_gt_20120407 = f['MRtarget_gt_20120407']
#         MRtarget_gt_20170922 = f['MRtarget_gt_20170922']
#         estMR_BR2 = f['estMR_BR2']
#         estMR_BR = f['estMR_BR']



#############################################################
#############################################################


# plt.figure(figsize=(18,18))
# plt.subplots_adjust(hspace=0.3, wspace=0.3)
# correct_blob1 = 0 #10000
# plt.subplot(321)
# plt.plot(threshes, (np.array(MR_S[::2])/grid2sqm)-correct_blob1, 'r-', lw=1, label='Est, MR, '+times[0])
# plt.axhline(y=(MRtarget_gt_20120407/overdig_factor), color='k', label='Obs, MR, '+times[0])

# correct_blob2 = 0 #45000
# plt.plot(threshes, (np.array(MR_S[1::2])/grid2sqm)-correct_blob2, 'r--', lw=2, label='Est, MR, '+times[1])
# plt.axhline(y=MRtarget_gt_20170922/overdig_factor, color='k', linestyle='--', label='Obs, MR, '+times[1])
# plt.axvline(x=.1, linestyle='--')
# plt.xlabel('Threshold wood probability')
# plt.ylabel(r"Wood, m$^2$"); #plt.ylim(7500,40000)
# plt.legend()
# plt.title('a)', loc='left')

# plt.subplot(322)
# plt.plot(threshes, (np.array(S[::2])/grid2sqm), 'r-', lw=1, label='Est, LR, '+times[0])
# plt.axhline(y=target_gt_20120407/overdig_factor, color='k', label='Obs, LR, '+times[0])

# correct_blob2 = 0 #55000
# plt.plot(threshes, (np.array(S[1::2])/grid2sqm)-correct_blob2, 'r--', lw=2, label='Est, LR, '+times[1])
# plt.axhline(y=target_gt_20170922/overdig_factor, color='k', linestyle='--', label='Obs, LR, '+times[1])
# plt.axvline(x=.1, linestyle='--')
# plt.xlabel('Threshold wood probability')
# plt.ylabel(r"Wood, m$^2$"); #plt.ylim(7500,35000)
# plt.legend()
# plt.title('b)', loc='left')

# plt.subplot(323)
# plt.plot(MR,np.cumsum(np.array(MR_BR)/overdig_factor/grid2sqm), 'k-', label='Obs, MR, '+times[0])
# plt.plot(MR,np.cumsum(np.array(MR_BR2)/overdig_factor/grid2sqm), 'k--', lw=2, label='Obs, MR, '+times[1])
# # plt.axhline(y=MRtarget_gt_20120407/overdig_factor, color='b', linestyle=':')
# # plt.axhline(y=MRtarget_gt_20170922/overdig_factor, color='b', linestyle=':')

# plt.plot(MR,np.cumsum(np.array(estMR_BR)/grid2sqm), 'r-', label='Est, MR, '+times[0])
# plt.plot(MR, np.cumsum(np.array(estMR_BR2)/grid2sqm), 'r--', lw=2, label='Est, MR, '+times[1])
# plt.title('c)', loc='left')
# plt.legend()
# plt.xlabel("Distance downstream (km)"); plt.ylabel(r"Wood, m$^2$")

# plt.subplot(324)
# plt.plot(LR,np.cumsum(np.array(BR)/overdig_factor/grid2sqm), 'k-', label='Obs, LR, '+times[0])
# plt.plot(LR,np.cumsum(np.array(BR2)/overdig_factor/grid2sqm), 'k--', lw=2, label='Obs, LR, '+times[1])
# # plt.axhline(y=target_gt_20120407/overdig_factor, color='b', linestyle=':')
# # plt.axhline(y=target_gt_20170922/overdig_factor, color='b', linestyle=':')

# plt.plot(LR,np.cumsum(np.array(estBR)/grid2sqm), 'r-', label='Est, LR, '+times[0])
# plt.plot(LR,np.cumsum(np.array(estBR2)/grid2sqm), 'r--', lw=2, label='Est, LR, '+times[1])
# plt.title('d) ', loc='left')
# plt.xlabel("Distance downstream (km)"); plt.ylabel(r"Wood, m$^2$")
# plt.legend()
# # plt.show()

# plt.subplot(325)
# plt.plot(np.cumsum(np.array(MR_BR)/overdig_factor/grid2sqm)[-1], np.cumsum(np.array(estMR_BR)/grid2sqm)[-1], 'ko', label='MR, 2012-04-07')
# plt.plot(np.cumsum(np.array(MR_BR2)/overdig_factor/grid2sqm)[-1], np.cumsum(np.array(estMR_BR2)/grid2sqm)[-1], 'ks', label='MR, 2017-09-22')
# plt.plot(np.cumsum(np.array(BR)/overdig_factor/grid2sqm)[-1], np.cumsum(np.array(estBR)/grid2sqm)[-1], 'ro', label='LR, 2012-04-07')
# plt.plot(np.cumsum(np.array(BR2)/overdig_factor/grid2sqm)[-1], np.cumsum(np.array(estBR2)/grid2sqm)[-1], 'rs', label='LR, 2017-09-22')
# yl=plt.xlim()
# plt.plot(yl, yl, 'b:', lw=2, label='1:1 relation')
# # plt.plot(yl, (yl[0], yl[1]*1.2), 'b:', lw=2, label='+20%')
# # plt.plot(yl, (yl[0], yl[1]*0.8), 'b:', lw=2, label='-20%')

# O = [MRtarget_gt_20120407,MRtarget_gt_20170922,target_gt_20120407,target_gt_20170922]
# E = [MRest_gt_20120407,MRest_gt_20170922,est_gt_20120407,est_gt_20170922]


# A = np.vstack([np.array(O)/overdig_factor, np.ones(len(O))]).T
# m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
# plt.plot(np.sort(np.array(O))/overdig_factor, m*np.sort(np.array(O))/overdig_factor + c, 'r:',lw=2, label='y = '+str(m)[:4]+'x+'+str(c)[:4])

# ### inverse prob
# A = np.vstack([np.array(E), np.ones(len(E))]).T
# m, c = np.linalg.lstsq(A, np.array(O)/overdig_factor, rcond=None)[0]
# print(m) 
# print(c)

# plt.legend()
# plt.ylabel(r"Estimated wood, m$^2$"); plt.xlabel(r"Observed wood, m$^2$")
# plt.title('e) ', loc='left')

# plt.subplot(326)
# x = np.cumsum(np.array(MR_BR)/overdig_factor/grid2sqm)
# y = np.cumsum(np.array(estMR_BR)/grid2sqm)
# plt.semilogy(MR, np.abs((x-y)/y)*100, 'k', label='MR, '+times[0])

# x = np.cumsum(np.array(MR_BR2)/overdig_factor/grid2sqm)
# y = np.cumsum(np.array(estMR_BR2)/grid2sqm)
# plt.plot(MR, np.abs((x-y)/y)*100, 'r-', label='MR, '+times[1])

# x = np.cumsum(np.array(BR)/overdig_factor/grid2sqm)
# y = np.cumsum(np.array(estBR)/grid2sqm)
# plt.plot(LR, np.abs((x-y)/y)*100, 'k--', label='LR, '+times[0])

# x = np.cumsum(np.array(BR2)/overdig_factor/grid2sqm)
# y = np.cumsum(np.array(estBR2)/grid2sqm)
# plt.plot(LR, np.abs((x-y)/y)*100, 'r--', label='LR, '+times[1])
# plt.axhline(y=20, color='b', linestyle=':', lw=2, label=r'20% error')
# plt.xlabel("Distance downstream (km)"); plt.ylabel(r"Percent error")
# plt.legend()
# plt.title('f) ', loc='left')

# # plt.show()

# plt.savefig("gt_wood_thres_analysis.png", dpi=300, bbox_inches="tight")
# plt.close()


# ################ large pieces

# plt.figure(figsize=(18,18))
# plt.subplots_adjust(hspace=0.3, wspace=0.3)
# correct_blob1 = 0 #10000
# plt.subplot(321)
# plt.plot(threshes, (np.array(MR_S_[::2])/grid2sqm)-correct_blob1, 'r-', lw=1, label='Est, MR, '+times[0])
# plt.axhline(y=(MRtarget_gt_20120407_/overdig_factor), color='k', label='Obs, MR, '+times[0])

# correct_blob2 = 0 #45000
# plt.plot(threshes, (np.array(MR_S_[1::2])/grid2sqm)-correct_blob2, 'r--', lw=2, label='Est, MR, '+times[1])
# plt.axhline(y=MRtarget_gt_20170922_/overdig_factor, color='k', linestyle='--', label='Obs, MR, '+times[1])
# plt.axvline(x=.1, linestyle='--')
# plt.xlabel('Threshold wood probability')
# plt.ylabel(r"Wood, m$^2$"); #plt.ylim(7500,40000)
# plt.legend()
# plt.title('a)', loc='left')

# plt.subplot(322)
# plt.plot(threshes, (np.array(S_[::2])/grid2sqm), 'r-', lw=1, label='Est, LR, '+times[0])
# plt.axhline(y=target_gt_20120407_/overdig_factor, color='k', label='Obs, LR, '+times[0])

# correct_blob2 = 0 #55000
# plt.plot(threshes, (np.array(S_[1::2])/grid2sqm)-correct_blob2, 'r--', lw=2, label='Est, LR, '+times[1])
# plt.axhline(y=target_gt_20170922_/overdig_factor, color='k', linestyle='--', label='Obs, LR, '+times[1])
# plt.axvline(x=.1, linestyle='--')
# plt.xlabel('Threshold wood probability')
# plt.ylabel(r"Wood, m$^2$"); #plt.ylim(7500,35000)
# plt.legend()
# plt.title('b)', loc='left')

# plt.subplot(323)
# plt.plot(MR,np.cumsum(np.array(MR_BR_)/overdig_factor/grid2sqm), 'k-', label='Obs, MR, '+times[0])
# plt.plot(MR,np.cumsum(np.array(MR_BR2_)/overdig_factor/grid2sqm), 'k--', lw=2, label='Obs, MR, '+times[1])
# # plt.axhline(y=MRtarget_gt_20120407_/overdig_factor, color='b', linestyle=':')
# # plt.axhline(y=MRtarget_gt_20170922_/overdig_factor, color='b', linestyle=':')

# plt.plot(MR,np.cumsum(np.array(estMR_BR_)/grid2sqm), 'r-', label='Est, MR, '+times[0])
# plt.plot(MR, np.cumsum(np.array(estMR_BR2_)/grid2sqm), 'r--', lw=2, label='Est, MR, '+times[1])
# plt.title('c)', loc='left')
# plt.legend()
# plt.xlabel("Distance downstream (km)"); plt.ylabel(r"Wood, m$^2$")

# plt.subplot(324)
# plt.plot(LR,np.cumsum(np.array(BR_)/overdig_factor/grid2sqm), 'k-', label='Obs, LR, '+times[0])
# plt.plot(LR,np.cumsum(np.array(BR2_)/overdig_factor/grid2sqm), 'k--', lw=2, label='Obs, LR, '+times[1])
# # plt.axhline(y=target_gt_20120407_/overdig_factor, color='b', linestyle=':')
# # plt.axhline(y=target_gt_20170922_/overdig_factor, color='b', linestyle=':')

# plt.plot(LR,np.cumsum(np.array(estBR_)/grid2sqm), 'r-', label='Est, LR, '+times[0])
# plt.plot(LR,np.cumsum(np.array(estBR2_)/grid2sqm), 'r--', lw=2, label='Est, LR, '+times[1])
# plt.title('d) ', loc='left')
# plt.xlabel("Distance downstream (km)"); plt.ylabel(r"Wood, m$^2$")
# plt.legend()
# # plt.show()

# plt.subplot(325)
# plt.plot(np.cumsum(np.array(MR_BR_)/overdig_factor/grid2sqm)[-1], np.cumsum(np.array(estMR_BR_)/grid2sqm)[-1], 'ko', label='MR, 2012-04-07')
# plt.plot(np.cumsum(np.array(MR_BR2_)/overdig_factor/grid2sqm)[-1], np.cumsum(np.array(estMR_BR2_)/grid2sqm)[-1], 'ks', label='MR, 2017-09-22')
# plt.plot(np.cumsum(np.array(BR_)/overdig_factor/grid2sqm)[-1], np.cumsum(np.array(estBR_)/grid2sqm)[-1], 'ro', label='LR, 2012-04-07')
# plt.plot(np.cumsum(np.array(BR2_)/overdig_factor/grid2sqm)[-1], np.cumsum(np.array(estBR2_)/grid2sqm)[-1], 'rs', label='LR, 2017-09-22')
# yl=plt.xlim()
# plt.plot(yl, yl, 'b:', lw=2, label='1:1 relation')
# # plt.plot(yl, (yl[0], yl[1]*1.2), 'b:', lw=2, label='+20%')
# # plt.plot(yl, (yl[0], yl[1]*0.8), 'b:', lw=2, label='-20%')

# O = [MRtarget_gt_20120407_,MRtarget_gt_20170922_,target_gt_20120407_,target_gt_20170922_]
# E = [MRest_gt_20120407_,MRest_gt_20170922_,est_gt_20120407_,est_gt_20170922_]


# A = np.vstack([np.array(O)/overdig_factor, np.ones(len(O))]).T
# m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
# plt.plot(np.sort(np.array(O))/overdig_factor, m*np.sort(np.array(O))/overdig_factor + c, 'r:',lw=2, label='y = '+str(m)[:4]+'x+'+str(c)[:4])

# ### inverse prob
# A = np.vstack([np.array(E), np.ones(len(E))]).T
# m, c = np.linalg.lstsq(A, np.array(O)/overdig_factor, rcond=None)[0]
# print(m) 
# print(c)

# plt.legend()
# plt.ylabel(r"Estimated wood, m$^2$"); plt.xlabel(r"Observed wood, m$^2$")
# plt.title('e) ', loc='left')

# plt.subplot(326)
# x = np.cumsum(np.array(MR_BR_)/overdig_factor/grid2sqm)
# y = np.cumsum(np.array(estMR_BR_)/grid2sqm)
# plt.semilogy(MR, np.abs((x-y)/y)*100, 'k', label='MR, '+times[0])

# x = np.cumsum(np.array(MR_BR2_)/overdig_factor/grid2sqm)
# y = np.cumsum(np.array(estMR_BR2_)/grid2sqm)
# plt.plot(MR, np.abs((x-y)/y)*100, 'r-', label='MR, '+times[1])

# x = np.cumsum(np.array(BR_)/overdig_factor/grid2sqm)
# y = np.cumsum(np.array(estBR_)/grid2sqm)
# plt.plot(LR, np.abs((x-y)/y)*100, 'k--', label='LR, '+times[0])

# x = np.cumsum(np.array(BR2_)/overdig_factor/grid2sqm)
# y = np.cumsum(np.array(estBR2_)/grid2sqm)
# plt.plot(LR, np.abs((x-y)/y)*100, 'r--', label='LR, '+times[1])
# plt.axhline(y=20, color='b', linestyle=':', lw=2, label=r'20% error')
# plt.xlabel("Distance downstream (km)"); plt.ylabel(r"Percent error")
# plt.legend()
# plt.title('f) ', loc='left')

# # plt.show()

# plt.savefig("gt_wood_largepieces_thres_analysis.png", dpi=300, bbox_inches="tight")
# plt.close()






### evaluate ability of model to capture change

## make transfer function between observed and estimated distributions











# plt.plot(MRtarget_gt_20120407/overdig_factor/grid2sqm, np.cumsum(np.array(estMR_BR)/grid2sqm)[-1], 'ko', label='MR, 2012-04-07')
# plt.plot(MRtarget_gt_20170922/overdig_factor/grid2sqm, np.cumsum(np.array(estMR_BR2)/grid2sqm)[-1], 'ks', label='MR, 2017-09-22')
# plt.plot(target_gt_20120407/overdig_factor/grid2sqm, np.cumsum(np.array(estBR)/grid2sqm)[-1], 'ro', label='LR, 2012-04-07')
# plt.plot(target_gt_20170922/overdig_factor/grid2sqm, np.cumsum(np.array(estBR2)/grid2sqm)[-1], 'rs', label='LR, 2017-09-22')



# #############################################################
# ### LR
# # get filtered wood probs, clipped to margins
# wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/Elwha_LR_*wood_filtered_bin0.1_regrid_final.tif'))
# print(len(wood_files))

# wood_files = [w for w in wood_files for t in times if t in w]

# print(wood_files)

# # Load in and concatenate all individual GeoTIFFs
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')

# # Rename the variable to a more useful name
# wood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

# #############################################################
# ### MR
# # get filtered wood probs, clipped to margins
# wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/Elwha_MR_*wood_filtered_bin0.1_regrid_final.tif'))
# print(len(wood_files))

# wood_files = [w for w in wood_files for t in times if t in w]
# print(wood_files)

# # Load in and concatenate all individual GeoTIFFs
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# geotiffs_ds = geotiffs_da.to_dataset('band')

# # Rename the variable to a more useful name
# MRwood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})











# # get timeaverage image for consistent lighting
# avim_ds = rioxarray.open_rasterio("../results/LR/LR_orthos_orig/Elwha_LR_im_time_mean_prob.tif", chunks=chunksize, dtype='uint8')
# avim_ds = avim_ds.to_dataset('band')
# print(avim_ds.dims)
# print(wood_geotiffs_ds.dims)

# #############################################################
# cmap=plt.cm.get_cmap('Blues', len(times))
# custom_palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

# for counter,g in tqdm(enumerate(geometries)):
#     print("Working on region {}".format(counter))

#     im_c = avim_ds.rio.clip([g], avim_ds.rio.crs)
#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)

#     tmp_da = xr.concat([im_c[1],im_c[2],im_c[3]],dim=('x','x','x'))
    
#     sum_da = np.zeros((wood_c.dims['x'],wood_c.dims['y']))
#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))

#         if inner_counter==0:
#             fig1, ax1 = plt.subplots()
#             plt.imshow(tmp_da.transpose()/255.)

#         wood_da = wood_c.wood.sel(time=time)

#         sum_da += wood_da.transpose().to_numpy()
#         ## keep overlaying contours with deeper and deeper color with time
#         CS1 = ax1.contour(wood_da.transpose(), colors=custom_palette[inner_counter])#, alpha=0.5)
#         # plt.axis('off')

#         # fmt = {}
#         # strs = [time]
#         # for l, s in zip(CS1.levels, strs):
#         #     fmt[l] = s

#         # # Label every other level using strings
#         # ax1.clabel(CS1, CS1.levels[1::2], inline=True, fmt=fmt, fontsize=10)

#     plt.axis('off')
#     plt.savefig(f"../results/LR/LR_wood/summary/wood_animation/Wood_01_frame_{counter}_alltime.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     del wood_da

#     fig1, ax1 = plt.subplots()
#     plt.imshow(tmp_da.transpose()/255.)
#     sum_da[sum_da==0] = np.nan
#     plt.imshow(100*(sum_da/len(times)), cmap='bwr')
#     plt.axis('off')
#     cb=plt.colorbar()
#     cb.set_label('Percent wood occupancy')
#     plt.savefig(f"../results/LR/LR_wood/summary/wood_animation/Wood_01_frame_{counter}_occupancy_alltime.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     del sum_da

#     tmp = wood_c.wood.sum("time", skipna=True)
#     tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/wood_animation/region{counter}/Elwha_LR_region_{counter}_wood_sum_time.tif", dtype=dtype)
#     del tmp

#     del tmp_da, im_c, wood_c


# #############################################################
# if run_bash:

#     # for i in range(len(geometries)):
#     #     try:
#     #         os.mkdir(f"../results/LR/LR_wood/summary/wood_animation/region{i}")
#     #     except:
#     #         pass
#     #     os.system(f'mv ../results/LR/LR_wood/summary/wood_animation/Wood_01_frame_{i}*.png ../results/LR/LR_wood/summary/wood_animation/region{i}')

#     #     os.system(f'convert -delay 100 ../results/LR/LR_wood/summary/wood_animation/region{i}/Wood_01_frame_{i}*occupancy*.png ../results/LR/LR_wood/summary/wood_animation/wood_animation_region{i}.gif')

#     ### run bash script to stitch region sums
#     os.chdir(f"../results/LR/LR_wood/summary")
#     os.system("bash mosaic_timesums.sh")
#     os.chdir(cwd)        



# #############################################################
# #############################################################
# ####################make time-difference rasters

# time0 = times[0]

# for counter,g in tqdm(enumerate(geometries)):
#     print("Working on region {}".format(counter))

#     wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
    
#     for inner_counter, time in enumerate(times[1:]):
#         print("Working on time {}".format(time))

#         if inner_counter==0:
#             wood_da0 = wood_c.wood.sel(time=time0)

#         wood_da = wood_c.wood.sel(time=time)

#         tmp = wood_da -  wood_da0

#         tmp.rio.to_raster(raster_path=f"../results/LR/LR_wood/summary/wood_animation/region{counter}/Elwha_LR_region_{counter}_wood_diff_time0.tif", dtype=dtype)
#         del tmp

#         del wood_da
#     del wood_c



# # #############################################################
# # #############################################################
# # ### whole-reach animation by time
# # tmp_da = xr.concat([avim_ds[1],avim_ds[2],avim_ds[3]],dim=('x','x','x'))

# # for inner_counter, time in enumerate(times):
# #     print("Working on time {}".format(time))

# #     if inner_counter==0:
# #         plt.imshow(tmp_da.transpose()/255.)

# #     wood_da = wood_geotiffs_ds.wood.sel(time=time)

# #     plt.contour(wood_da.transpose(), colors=custom_palette[inner_counter], alpha=0.5)

# # plt.axis('off')
# # plt.savefig(f"../results/LR/LR_wood/summary/wood_animation/Wood_LR_frame_{counter}_allreach_alltime.png", dpi=300, bbox_inches='tight')
# # plt.close()
# # del wood_da
# # del tmp_da

# # #############################################################
# # if run_bash:

# #     for i in range(len(geometries)):
# #         try:
# #             os.mkdir(f"../results/LR/LR_wood/summary/region{i}")
# #         except:
# #             pass
# #         os.system(f'mv ../results/LR/LR_wood/summary/Wood_01_frame_{i}*.png ../results/LR/LR_wood/summary/wood_animation/region{i}')

# #         os.system(f'convert -delay 100 ../results/LR/LR_wood/summary/wood_animation/region{i}/Wood_01_frame_{i}*.png wood_animation_region{i}.gif')
