
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

cwd = os.getcwd()

# Create variable used for time axis
time_var = xr.Variable('time',times)

# ## start client
# client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)


#############################################################
#########################################################
# fpoints = sorted(glob('../raw_data/GIS/Sep22_2017/pts_on_braid_epsg6339.geojson'))
# with open(fpoints[0]) as f:
#     gj = json.load(f)
# features = gj['features']

# points = [f['geometry']['coordinates'] for f in features]
# print("{} sample points".format(len(points)))

fpoints = sorted(glob('../raw_data/GIS/Sep22_2017/dem_pts_braids.geojson'))
with open(fpoints[0]) as f:
    gj = json.load(f)
features = gj['features']

points = [f['geometry']['coordinates'] for f in features]
z = [f['properties']['Elwha_LR_20170922_DEM_regrid_1'] for f in features]
print("{} sample points".format(len(z)))

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


x=np.array(points)[:,0]
y=np.array(points)[:,1]

print(len(x))



# pdem = dem_geotiffs_ds.dem.sel(x=x,y=y, method="nearest").to_numpy()

# np.savez('DEM_braid_elev_pts.npz', x = x, y = y, times=times, pdem = pdem)

# pdem = []
# for (xx,yy) in tqdm(zip(x,y)):
#     tmp = dem_geotiffs_ds.dem.sel(x=xx,y=yy, method="nearest", tolerance=10).to_numpy()
#     print(tmp)
#     pdem.append(tmp)

# with np.load('DEM_braid_elev_pts.npz', allow_pickle=True) as f:
#     x = f['x']
#     y = f['y']
#     pdem = f['pdem']


import scipy.linalg

# regular grid covering the domain of the data
X,Y = np.meshgrid(dem_geotiffs_ds.x.to_numpy(), dem_geotiffs_ds.y.to_numpy())

order = 1    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[x, y, np.ones(len(x))]
    C,_,_,_ = scipy.linalg.lstsq(A, z)    # coefficients
    
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]

elif order == 2:
    XX = X.flatten()
    YY = Y.flatten()
    # best-fit quadratic curve
    A = np.c_[np.ones(len(x)), np.vstack((x,y)), np.prod(np.vstack((x,y)) axis=1), np.vstack((x,y))**2]
    C,_,_,_ = scipy.linalg.lstsq(A, z)
    
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)    


for time in times:

    tmp = dem_geotiffs_ds.dem.sel(time=time)
    tmp.data = tmp.data - Z

    tmp.rio.to_raster(raster_path=f"DEM_detrend.tif", dtype=dtype)
    del tmp
