## Dan Buscombe, Marda Science
## Apr, 2023
##
## Does this:
## 1. 

## Where are we in the sequence?

import os, json
import rioxarray
import xarray as xr 
from glob import glob 
import matplotlib.pyplot as plt
from dask.distributed import Client
from tqdm import tqdm
# import matplotlib.colors
import numpy as np
# from scipy import ndimage

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

n_workers = 20
threads_per_worker = 2
memory_limit='100GB'

cwd = os.getcwd()
# run_bash = True

## factor that converts grid uints 1/8 x 1/8
# into units 1 x 1, i.e. 8 x 8
grid2sqm = 64

## we estimate over-ditizization factor
# overdig_factor = 1.5 
overdig_factor = 1.2


#############################################################
## start client
client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

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


brfile = '../results/LR/LR_wood/wood_detect/LR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
budget_reaches = gj['features']

brfile = '../results/MR/MR_wood/wood_detect/MR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
MRbudget_reaches = gj['features']

gcfile = '../results/LR/LR_wood/wood_detect/LR_global_clipper.geojson'
with open(gcfile) as f:
    gj = json.load(f)
global_clipper = gj['features']

budget_reaches_redo = []
for b in budget_reaches:
    budget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))

MRbudget_reaches_redo = []
for b in MRbudget_reaches:
    MRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))


#############################################################
#############################################################

#############################################################
### LR
# get filtered wood probs, clipped to margins
wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/Elwha_LR_*wood_filtered_prob_regrid.tif'))
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

#############################################################
### MR
# get filtered wood probs, clipped to margins
wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/Elwha_MR_*wood_filtered_prob_regrid.tif'))
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

#############################################################
### LR

gt_20170922 = rioxarray.open_rasterio("../raw_data/dig_wood/LR_20170922_dig_wood_clipped_active_budgetextent.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
gt_20120407 = rioxarray.open_rasterio("../raw_data/dig_wood/LR_20120407_dig_wood_clipped_active_budgetextent.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

### sum of all wood pixels is the target metric
target_gt_20170922 = gt_20170922[1].sum().compute().to_numpy()
target_gt_20120407 = gt_20120407[1].sum().compute().to_numpy()

threshes = np.arange(.05,.5,.01)
S=[]
for thres in threshes:
    for time in times:
        result = (wood_geotiffs_ds.wood.sel(time=time)>thres).sum().compute().to_numpy()
        print(f"{thres}: {result}")
        S.append(float(result))


LR_20170922 = rioxarray.open_rasterio("../results/LR/LR_wood/wood_detect/Elwha_LR_2017-09-22_wood_filtered_bin0.1_regrid_final.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
LR_20120407 = rioxarray.open_rasterio("../results/LR/LR_wood/wood_detect/Elwha_LR_2012-04-07_wood_filtered_bin0.1_regrid_final.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

# # ### sum of all wood pixels is the target metric
# est_gt_20170922 = LR_20170922[1].sum().compute().to_numpy()
# est_gt_20120407 = LR_20120407[1].sum().compute().to_numpy()

#############################################################
### MR

MRgt_20170922 = rioxarray.open_rasterio("../raw_data/dig_wood/MR_20170922_dig_wood_clipped_active_budgetextent.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
MRgt_20120407 = rioxarray.open_rasterio("../raw_data/dig_wood/MR_20120407_dig_wood_clipped_active_budgetextent_v2.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

# ### sum of all wood pixels is the target metric
# MRtarget_gt_20170922 = MRgt_20170922[1].sum().compute().to_numpy()
# MRtarget_gt_20120407 = MRgt_20120407[1].sum().compute().to_numpy()

threshes = np.arange(.05,.5,.01)
MR_S=[]
for thres in threshes:
    for time in times:
        result = (MRwood_geotiffs_ds.wood.sel(time=time)>thres).sum().compute().to_numpy()
        print(f"{thres}: {result}")
        MR_S.append(float(result))

MR_20170922 = rioxarray.open_rasterio("../results/MR/MR_wood/wood_detect/Elwha_MR_2017-09-22_wood_filtered_bin0.1_regrid_final.tif", chunks=chunksize, dtype=dtype).to_dataset('band')
MR_20120407 = rioxarray.open_rasterio("../results/MR/MR_wood/wood_detect/Elwha_MR_2012-04-07_wood_filtered_bin0.1_regrid_final.tif", chunks=chunksize, dtype=dtype).to_dataset('band')

# # ### sum of all wood pixels is the target metric
# MRest_gt_20170922 = MR_20170922[1].sum().compute().to_numpy()
# MRest_gt_20120407 = MR_20120407[1].sum().compute().to_numpy()


#############################################################
#############################################################


BR=[]
for g in tqdm(budget_reaches_redo):
    wood_gt = gt_20120407.rio.clip([g], gt_20120407.rio.crs)
    result = (wood_gt[1]).sum().compute().to_numpy() 
    BR.append(float(result))

BR2=[]
for g in tqdm(budget_reaches_redo):
    wood_gt = gt_20170922.rio.clip([g], gt_20170922.rio.crs)
    result = (wood_gt[1]).sum().compute().to_numpy() 
    BR2.append(float(result))


MR_BR=[]
for g in tqdm(MRbudget_reaches_redo):
    wood_gt = MRgt_20120407.rio.clip([g], MRgt_20120407.rio.crs)
    result = (wood_gt[1]).sum().compute().to_numpy() 
    MR_BR.append(float(result))

MR_BR2=[]
for g in tqdm(MRbudget_reaches_redo):
    wood_gt = MRgt_20170922.rio.clip([g], MRgt_20170922.rio.crs)
    result = (wood_gt[1]).sum().compute().to_numpy() 
    MR_BR2.append(float(result))

#################


estBR=[]
for g in tqdm(budget_reaches_redo):
    wood_gt = LR_20120407.rio.clip([g], LR_20120407.rio.crs)
    result = (wood_gt[1]).sum().compute().to_numpy() 
    estBR.append(float(result))

estBR2=[]
for g in tqdm(budget_reaches_redo):
    wood_gt = LR_20170922.rio.clip([g], LR_20170922.rio.crs)
    result = (wood_gt[1]).sum().compute().to_numpy() 
    estBR2.append(float(result))

estMR_BR=[]
for g in tqdm(MRbudget_reaches_redo):
    wood_gt = MR_20120407.rio.clip([g], MR_20120407.rio.crs)
    result = (wood_gt[1]).sum().compute().to_numpy() 
    estMR_BR.append(float(result))

estMR_BR2=[]
for g in tqdm(MRbudget_reaches_redo):
    wood_gt = MR_20170922.rio.clip([g], MR_20170922.rio.crs)
    result = (wood_gt[1]).sum().compute().to_numpy() 
    estMR_BR2.append(float(result))


MRtarget_gt_20120407 = np.cumsum(np.array(MR_BR)/grid2sqm)[-1]
MRtarget_gt_20170922 = np.cumsum(np.array(MR_BR2)/grid2sqm)[-1]

target_gt_20120407 = np.cumsum(np.array(BR)/grid2sqm)[-1]
target_gt_20170922 = np.cumsum(np.array(BR2)/grid2sqm)[-1]


MRest_gt_20120407 = np.cumsum(np.array(estMR_BR)/grid2sqm)[-1]
MRest_gt_20170922 = np.cumsum(np.array(estMR_BR2)/grid2sqm)[-1]

est_gt_20120407 = np.cumsum(np.array(estBR)/grid2sqm)[-1]
est_gt_20170922 = np.cumsum(np.array(estBR2)/grid2sqm)[-1]


print(f"Obs: 2012-04-07: {MRtarget_gt_20120407/overdig_factor}")
print(f"Est: 2012-04-07: {MRest_gt_20120407}")

print(f"Obs: 2017-09-22: {MRtarget_gt_20170922/overdig_factor}")
print(f"Est: 2017-09-22: {MRest_gt_20170922}")

print(100*((MRtarget_gt_20120407/overdig_factor)-MRest_gt_20120407)/MRest_gt_20120407)
print(100*((MRtarget_gt_20170922/overdig_factor)-MRest_gt_20170922)/MRest_gt_20170922)

print(f"Obs: 2012-04-07: {target_gt_20120407/overdig_factor}")
print(f"Est: 2012-04-07: {est_gt_20120407}")

print(f"Obs: 2017-09-22: {target_gt_20170922/overdig_factor}")
print(f"Est: 2017-09-22: {est_gt_20170922}")

print(100*((target_gt_20120407/overdig_factor)-est_gt_20120407)/est_gt_20120407)
print(100*((target_gt_20170922/overdig_factor)-est_gt_20170922)/est_gt_20170922)


np.savez('MR_eval_wood_budget.npz', MRtarget_gt_20120407 = target_gt_20120407, MRtarget_gt_20170922=target_gt_20170922, estMR_BR2=estMR_BR2, estMR_BR=estMR_BR, MRbudget_reaches=MRbudget_reaches_redo, grid2sqm=grid2sqm, overdig_factor=overdig_factor)

np.savez('LR_eval_wood_budget.npz', LRtarget_gt_20120407 = target_gt_20120407, LRtarget_gt_20170922=target_gt_20170922, estLR_BR2=estBR2, estLR_BR=estBR, LRbudget_reaches=budget_reaches_redo, grid2sqm=grid2sqm, overdig_factor=overdig_factor)

#############################################################
#############################################################


plt.figure(figsize=(18,18))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
# correct_blob1 = -10000
plt.subplot(321)
plt.plot(threshes, (np.array(MR_S[::2])/grid2sqm), 'r-', lw=1, label='Est, MR, '+times[0])
plt.axhline(y=(MRtarget_gt_20120407/overdig_factor), color='k', label='Obs, MR, '+times[0])

correct_blob2 = 45000
plt.plot(threshes, (np.array(MR_S[1::2])/grid2sqm)-correct_blob2, 'r--', lw=2, label='Est, MR, '+times[1])
plt.axhline(y=MRtarget_gt_20170922/overdig_factor, color='k', linestyle='--', label='Obs, MR, '+times[1])
plt.axvline(x=.1, linestyle='--')
plt.xlabel('Threshold wood probability')
plt.ylabel(r"Wood, m$^2$"); #plt.ylim(7500,40000)
plt.legend()
plt.title('a)', loc='left')

plt.subplot(322)
plt.plot(threshes, (np.array(S[::2])/grid2sqm), 'r-', lw=1, label='Est, LR, '+times[0])
plt.axhline(y=target_gt_20120407/overdig_factor, color='k', label='Obs, LR, '+times[0])

correct_blob2 = 55000
plt.plot(threshes, (np.array(S[1::2])/grid2sqm)-correct_blob2, 'r--', lw=2, label='Est, LR, '+times[1])
plt.axhline(y=target_gt_20170922/overdig_factor, color='k', linestyle='--', label='Obs, LR, '+times[1])
plt.axvline(x=.1, linestyle='--')
plt.xlabel('Threshold wood probability')
plt.ylabel(r"Wood, m$^2$"); #plt.ylim(7500,35000)
plt.legend()
plt.title('b)', loc='left')

plt.subplot(323)
plt.plot(np.cumsum(np.array(MR_BR)/overdig_factor/grid2sqm), 'k-', label='Obs, MR, '+times[0])
plt.plot(np.cumsum(np.array(MR_BR2)/overdig_factor/grid2sqm), 'k--', lw=2, label='Obs, MR, '+times[1])
# plt.axhline(y=MRtarget_gt_20120407/overdig_factor, color='b', linestyle=':')
# plt.axhline(y=MRtarget_gt_20170922/overdig_factor, color='b', linestyle=':')

plt.plot(np.cumsum(np.array(estMR_BR)/grid2sqm), 'r-', label='Est, MR, '+times[0])
plt.plot(np.cumsum(np.array(estMR_BR2)/grid2sqm), 'r--', lw=2, label='Est, MR, '+times[1])
plt.title('c)', loc='left')
plt.legend()
plt.xlabel("Accounting reach (increasing downstream)"); plt.ylabel(r"Wood, m$^2$")

plt.subplot(324)
plt.plot(np.cumsum(np.array(BR)/overdig_factor/grid2sqm), 'k-', label='Obs, LR, '+times[0])
plt.plot(np.cumsum(np.array(BR2)/overdig_factor/grid2sqm), 'k--', lw=2, label='Obs, LR, '+times[1])
# plt.axhline(y=target_gt_20120407/overdig_factor, color='b', linestyle=':')
# plt.axhline(y=target_gt_20170922/overdig_factor, color='b', linestyle=':')

plt.plot(np.cumsum(np.array(estBR)/grid2sqm), 'r-', label='Est, LR, '+times[0])
plt.plot(np.cumsum(np.array(estBR2)/grid2sqm), 'r--', lw=2, label='Est, LR, '+times[1])
plt.title('d) ', loc='left')
plt.xlabel("Accounting reach (increasing downstream)"); plt.ylabel(r"Wood, m$^2$")
plt.legend()
# plt.show()

plt.subplot(325)
plt.plot(MRtarget_gt_20120407/overdig_factor, np.cumsum(np.array(estMR_BR)/grid2sqm)[-1], 'ko', label='MR, 2012-04-07')
plt.plot(MRtarget_gt_20170922/overdig_factor, np.cumsum(np.array(estMR_BR2)/grid2sqm)[-1], 'ks', label='MR, 2017-09-22')
plt.plot(target_gt_20120407/overdig_factor, np.cumsum(np.array(estBR)/grid2sqm)[-1], 'ro', label='LR, 2012-04-07')
plt.plot(target_gt_20170922/overdig_factor, np.cumsum(np.array(estBR2)/grid2sqm)[-1], 'rs', label='LR, 2017-09-22')
yl=plt.xlim()
plt.plot(yl, yl, 'b:', lw=2, label='1:1 relation')
# plt.plot(yl, (yl[0], yl[1]*1.2), 'b:', lw=2, label='+20%')
# plt.plot(yl, (yl[0], yl[1]*0.8), 'b:', lw=2, label='-20%')

O = [MRtarget_gt_20120407,MRtarget_gt_20170922,target_gt_20120407,target_gt_20170922]
E = [np.cumsum(np.array(estMR_BR)/grid2sqm)[-1],
     np.cumsum(np.array(estMR_BR2)/grid2sqm)[-1],
     np.cumsum(np.array(estBR)/grid2sqm)[-1],
     np.cumsum(np.array(estBR2)/grid2sqm)[-1]
     ]

A = np.vstack([np.array(O)/overdig_factor, np.ones(len(O))]).T
m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
plt.plot(np.sort(np.array(O))/overdig_factor, m*np.sort(np.array(O))/overdig_factor + c, 'r:',lw=2, label='y = '+str(m)[:4]+'x+'+str(c)[:4])

### inverse prob
A = np.vstack([np.array(E), np.ones(len(E))]).T
m, c = np.linalg.lstsq(A, np.array(O)/overdig_factor, rcond=None)[0]
print(m) 
print(c)
# 1.168096131577042
# -6183.420519070709
# plt.plot(np.array(O)/overdig_factor, m*np.array(E)+c, 'mp', label='Corrected')

plt.legend()
plt.ylabel(r"Estimated wood, m$^2$"); plt.xlabel(r"Observed wood, m$^2$")
plt.title('e) ', loc='left')

plt.subplot(326)
x = np.cumsum(np.array(MR_BR)/overdig_factor/grid2sqm)
y = np.cumsum(np.array(estMR_BR)/grid2sqm)
plt.plot(np.abs((x-y)/y)*100, 'k', label='MR, '+times[0])

x = np.cumsum(np.array(MR_BR2)/overdig_factor/grid2sqm)
y = np.cumsum(np.array(estMR_BR2)/grid2sqm)
plt.plot(np.abs((x-y)/y)*100, 'r-', label='MR, '+times[1])

x = np.cumsum(np.array(BR)/overdig_factor/grid2sqm)
y = np.cumsum(np.array(estBR)/grid2sqm)
plt.plot(np.abs((x-y)/y)*100, 'k--', label='LR, '+times[0])

x = np.cumsum(np.array(BR2)/overdig_factor/grid2sqm)
y = np.cumsum(np.array(estBR2)/grid2sqm)
plt.plot(np.abs((x-y)/y)*100, 'r--', label='LR, '+times[1])
plt.axhline(y=20, color='b', linestyle=':', lw=2, label=r'20% error')
plt.xlabel("Accounting reach (increasing downstream)"); plt.ylabel(r"Percent error")
plt.legend()
plt.title('f) ', loc='left')

# plt.show()

plt.savefig("gt_wood_thres_analysis.png", dpi=300, bbox_inches="tight")
plt.close()


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
