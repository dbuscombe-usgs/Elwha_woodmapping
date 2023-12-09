
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

import scipy.linalg
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
# ## start client
client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

cwd = os.getcwd()

# Create variable used for time axis
time_var = xr.Variable('time',times)


#############################################################
#########################################################


####################################################################
############################## LR
####################################################################
####################################################################

dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/*LR_*DEM_regrid.tif'))
print(len(dem_files))

#############################################################

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
dem_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})

dem_geotiffs_ds = dem_geotiffs_ds.drop_vars(2)
print(dem_geotiffs_ds.to_array().shape)

#############################################################
#########################################################

####========================================================

## make a new file using the Point Sampling Tool in QGIS - load the briad shapefile, and the DEM
fpoints = sorted(glob('../raw_data/GIS/Sep22_2017/dem_pts_braids.geojson'))
with open(fpoints[0]) as f:
    gj = json.load(f)
features = gj['features']

points = [f['geometry']['coordinates'] for f in features]
z = [f['properties']['Elwha_LR_20170922_DEM_regrid_1'] for f in features]
print("{} sample points".format(len(z)))

x=np.array(points)[:,0]
y=np.array(points)[:,1]

print(len(x))

zLR=sorted(z)


## https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
# regular grid covering the domain of the data
X,Y = np.meshgrid(dem_geotiffs_ds.x.to_numpy(), dem_geotiffs_ds.y.to_numpy())

# # best-fit linear plane
# A = np.c_[x, y, np.ones(len(x))]
# C,_,_,_ = scipy.linalg.lstsq(A, z)    # coefficients

# # evaluate it on grid
# Z = C[0]*X + C[1]*Y + C[2]
# Z = Z - Z.min()
# del A, C, X, Y, x, y, points

# regular grid covering the domain of the data
XX = X.flatten()
YY = Y.flatten()
del Y

# best-fit quadratic curve
A = np.c_[np.ones(len(x)), np.vstack((x,y)).T, np.prod(np.vstack((x,y)).T, axis=1), np.vstack((x,y)).T**2]
C,_,_,_ = scipy.linalg.lstsq(A, z)

# evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
del X
del A, C, x, y, points
del YY
# time = '2017-09-22'
offset = 10


time='2013-09-19'
tmp = dem_geotiffs_ds.dem.sel(time=time).to_numpy()
tmp = tmp - tmp.min()
tmp = (offset+tmp)-Z
tmp[tmp<0] = np.nan
tmp = tmp - offset 
plt.imshow(tmp); plt.colorbar(); plt.axis('off'); plt.savefig('DEM_detrend.png',dpi=300, bbox_inches='tight'); plt.close()




for time in times[:8]:
        
    tmp = dem_geotiffs_ds.dem.sel(time=time).persist()
    tmp.data = tmp.data - tmp.data.min()
    tmp.data = (offset + tmp.data) - Z
    tmp.data[tmp.data<=0] = np.nan
    tmp.data = tmp.data - offset
    tmp = tmp.where(~np.isnan(dem_geotiffs_ds.dem.sel(time=time)))
    tmp = tmp.where(dem_geotiffs_ds.dem.sel(time=time)>0)

    tmp.rio.to_raster(raster_path=f"../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/LR_DEM_detrend_"+time+".tif", dtype=dtype)
    del tmp


for time in times[8:]:
        
    tmp = dem_geotiffs_ds.dem.sel(time=time).persist()
    tmp.data = tmp.data - tmp.data.min()
    tmp.data = (offset + tmp.data) - Z
    tmp.data[tmp.data<=0] = np.nan
    tmp.data = tmp.data - offset
    tmp = tmp.where(~np.isnan(dem_geotiffs_ds.dem.sel(time=time)))
    tmp = tmp.where(dem_geotiffs_ds.dem.sel(time=time)>0)

    tmp.rio.to_raster(raster_path=f"../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/LR_DEM_detrend_"+time+".tif", dtype=dtype)
    del tmp



########################### make LR average detrended DEM
###############################################################################################

dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/LR_DEM_detrend_2*.tif'))
print(len(dem_files))

######### get regions 
regions = sorted(glob('../raw_data/GIS/LR*ID*_epsg6339.geojson'))
regions = [r for r in regions if 'pts' not in r]
print("{} regions".format(len(regions)))

geometries = []
for r in regions:
    with open(r) as f:
        gj = json.load(f)
    features = gj['features'][0]

    geometries.append(features['geometry'])

#############################################################

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
dem_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})

print(dem_geotiffs_ds.to_array().shape)

dem_min = dem_geotiffs_ds.dem.min().compute().to_numpy()
print(dem_min)

for counter,g in tqdm(enumerate(geometries)):
    dem_c = dem_geotiffs_ds.rio.clip([g], dem_geotiffs_ds.rio.crs)

    tmp = dem_c.dem.mean("time", skipna=True).persist()

    tmp.data = tmp.data - dem_min

    tmp.rio.to_raster(raster_path=f"../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/LR_DEM_detrend_global_region_{counter}.tif", dtype=dtype)
    del tmp

cwd = os.getcwd()
os.chdir("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016")

os.system("gdalbuildvrt -input_file_list alltifs.txt mosaic.vrt")
os.remove('LR_DEM_detrend_global.tif')
os.system('gdal_translate -co "COMPRESS=LZW" mosaic.vrt LR_DEM_detrend_global.tif')
os.remove('LR_DEM_detrend_global_regrid.tif')
os.system('gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 LR_DEM_detrend_global.tif LR_DEM_detrend_global_regrid.tif')

os.system('rm LR*global_region*.tif')

os.chdir(cwd)



####################################################################
############################## MR
####################################################################
####################################################################

dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/*MR_*DEM_regrid.tif'))
print(len(dem_files))

#############################################################

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
dem_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})

dem_geotiffs_ds = dem_geotiffs_ds.drop_vars(2)
print(dem_geotiffs_ds.to_array().shape)


####========================================================

## make a new file using the Point Sampling Tool in QGIS - load the briad shapefile, and the DEM
fpoints = sorted(glob('../raw_data/GIS/Sep22_2017/MR_dem_pts_braids_epsg6339.geojson'))
with open(fpoints[0]) as f:
    gj = json.load(f)
features = gj['features']

points = [f['geometry']['coordinates'] for f in features]
z = [f['properties']['Elwha_MR_20170922_DEM_regrid_1'] for f in features]
print("{} sample points".format(len(z)))

x=np.array(points)[:,0]
y=np.array(points)[:,1]

print(len(x))

zMR=sorted(z)


dists = pd.read_csv('br_dists.csv')
LR = np.hstack((0,np.array(dists['LR'])))
MR = np.hstack((0,np.array(dists['MR'][:43])))
## rescale distances


def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    Code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn


LR = rescale_array(LR,11,2)

MR = rescale_array(MR[::-1],12,20)


zLR = np.array(zLR)

zMR = np.array(zMR)
# zMR = zMR[zMR<25]
# zMR = zMR[zMR>1]

from matplotlib.patches import Rectangle

import geopandas as gpd

file = '../raw_data/GIS/LR_active_widths.geojson'
LR_widths = gpd.read_file(file)
LR_widths = LR_widths['length'].values


file = '../raw_data/GIS/MR_active_widths.geojson'
MR_widths = gpd.read_file(file)
MR_widths = MR_widths['length'].values




plt.figure(figsize=(18,6))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.subplot(221)
# plt.plot(np.linspace(0,MR[-1],len(zMR)), (sorted(zMR)[::-1]-np.max(zMR))/1000,'k',label='MR')
# plt.plot(np.linspace(0,LR[-1],len(zLR)), (sorted(zLR)[::-1]-np.max(zLR))/1000,'r--', lw=2, label='LR')

rec=Rectangle((11,0), 1, 130, clip_on=False, color='gray')
plt.gca().add_artist(rec)

plt.plot(np.linspace(MR[0],MR[-1],len(zMR)), np.array(sorted(zMR)[::-1]),'k',label='MR') #-np.max(zMR)
plt.plot(np.linspace(LR[0],LR[-1],len(zLR)), np.array(sorted(zLR)[::-1]),'r--', lw=2, label='LR') #-np.max(zLR)
# yl=plt.ylim()

# O = np.linspace(0,MR[-1],len(zMR))
# E = (sorted(zMR)[::-1]-np.max(zMR))/1000
# A = np.vstack([np.array(O), np.ones(len(O))]).T
# m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
# plt.plot(np.sort(np.array(O)), m*np.sort(np.array(O)) + c, 'k:',lw=2, label='y = '+str(m)[:8]+'x+'+str(c)[:8])

# O = np.linspace(0,LR[-1],len(zLR))
# E = (sorted(zLR)[::-1]-np.max(zLR))/1000
# A = np.vstack([np.array(O), np.ones(len(O))]).T
# m, c = np.linalg.lstsq(A, np.array(E), rcond=None)[0]
# plt.plot(np.sort(np.array(O)), m*np.sort(np.array(O)) + c, 'r:',lw=2, label='y = '+str(m)[:8]+'x+'+str(c)[:8])

plt.title('c) ', loc='left')
plt.ylabel('Elevation (m; NAVD88)'); plt.xlabel('River kilometer')
plt.legend()
plt.gca().invert_xaxis()
plt.ylim(0,130)
plt.text(11,100,'former\nLake\nAldwell')


plt.subplot(222)

rec=Rectangle((11,0), 1, 600, clip_on=False, color='gray')
plt.gca().add_artist(rec)
plt.plot(np.linspace(MR[0],MR[-1],len(MR_widths)), MR_widths,'k',label='MR') #-np.max(zMR)
plt.plot(np.linspace(LR[0],LR[-1],len(LR_widths)), LR_widths,'r--', lw=2, label='LR') #-np.max(zLR)

plt.title('d) ', loc='left')
plt.ylabel('Maximum active\nchannel width (m)'); plt.xlabel('River kilometer')
plt.legend()
plt.gca().invert_xaxis()
plt.ylim(0,600)
plt.text(11,450,'former\nLake\nAldwell')

plt.savefig("Elev_width_profiles.png", dpi=300, bbox_inches="tight")
plt.close()



## https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
# regular grid covering the domain of the data
X,Y = np.meshgrid(dem_geotiffs_ds.x.to_numpy(), dem_geotiffs_ds.y.to_numpy())

# # best-fit linear plane
# A = np.c_[x, y, np.ones(len(x))]
# C,_,_,_ = scipy.linalg.lstsq(A, z)    # coefficients

# # evaluate it on grid
# Z = C[0]*X + C[1]*Y + C[2]
# Z = Z - Z.min()
# del A, C, X, Y, x, y, points

# regular grid covering the domain of the data
XX = X.flatten()
YY = Y.flatten()
del Y
# best-fit quadratic curve
A = np.c_[np.ones(len(x)), np.vstack((x,y)).T, np.prod(np.vstack((x,y)).T, axis=1), np.vstack((x,y)).T**2]
C,_,_,_ = scipy.linalg.lstsq(A, zMR)

# evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
del X
del A, C, x, y, points

offset = 10

for time in times[:8]:
        
    tmp = dem_geotiffs_ds.dem.sel(time=time).persist()
    tmp.data = tmp.data - tmp.data.min()
    tmp.data = (offset + tmp.data) - Z
    tmp.data[tmp.data<=0] = np.nan
    tmp.data = tmp.data - offset
    tmp = tmp.where(~np.isnan(dem_geotiffs_ds.dem.sel(time=time)))
    tmp = tmp.where(dem_geotiffs_ds.dem.sel(time=time)>0)

    tmp.rio.to_raster(raster_path=f"../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_"+time+".tif", dtype=dtype)
    del tmp

for time in times[8:]:
        
    tmp = dem_geotiffs_ds.dem.sel(time=time).persist()
    tmp.data = tmp.data - tmp.data.min()
    tmp.data = (offset + tmp.data) - Z
    tmp.data[tmp.data<=0] = np.nan
    tmp.data = tmp.data - offset
    tmp = tmp.where(~np.isnan(dem_geotiffs_ds.dem.sel(time=time)))
    tmp = tmp.where(dem_geotiffs_ds.dem.sel(time=time)>0)

    tmp.rio.to_raster(raster_path=f"../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_"+time+".tif", dtype=dtype)
    del tmp


########################### make MR average detrended DEM
###############################################################################################

dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_2*.tif'))
print(len(dem_files))

######### get regions 
regions = sorted(glob('../raw_data/GIS/MR*ID*_epsg6339.geojson'))
regions = [r for r in regions if 'pts' not in r]
print("{} regions".format(len(regions)))

geometries = []
for r in regions:
    with open(r) as f:
        gj = json.load(f)
    features = gj['features'][0]

    geometries.append(features['geometry'])

#############################################################

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
dem_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})

print(dem_geotiffs_ds.to_array().shape)

dem_min = dem_geotiffs_ds.dem.min().compute().to_numpy()
print(dem_min)

for counter,g in tqdm(enumerate(geometries)):
    dem_c = dem_geotiffs_ds.rio.clip([g], dem_geotiffs_ds.rio.crs)

    tmp = dem_c.dem.mean("time", skipna=True).persist()

    tmp.data = tmp.data - dem_min

    tmp.rio.to_raster(raster_path=f"../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/MR_DEM_detrend_global_region_{counter}.tif", dtype=dtype)
    del tmp

cwd = os.getcwd()
os.chdir("../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016")

os.system("gdalbuildvrt -input_file_list MR_alltifs.txt mosaic.vrt")
os.remove('MR_DEM_detrend_global.tif')
os.system('gdal_translate -co "COMPRESS=LZW" mosaic.vrt MR_DEM_detrend_global.tif')
os.remove('MR_DEM_detrend_global_regrid.tif')
os.system('gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 MR_DEM_detrend_global.tif MR_DEM_detrend_global_regrid.tif')
os.system('rm MR*global_region*.tif')

os.chdir(cwd)
