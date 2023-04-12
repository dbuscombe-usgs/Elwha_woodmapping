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


## Where are we in the sequence?
## 1.filter_wood_by_av_veg_water_dev.py
## 2. make_wood_movies.py
## 3. >>>> timeseries_analysis.py

import json, os
import rioxarray
import xarray as xr 
from glob import glob 
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client
from tqdm import tqdm
from datetime import datetime
from scipy.stats import pearsonr


## https://climate-cms.org/posts/2019-07-29-multi-apply-along-axis.html
def multi_apply_along_axis(func1d, axis, arrs, *args, **kwargs):
    """
    Given a function `func1d(A, B, C, ..., *args, **kwargs)`  that acts on 
    multiple one dimensional arrays, apply that function to the N-dimensional
    arrays listed by `arrs` along axis `axis`
    
    If `arrs` are one dimensional this is equivalent to::
    
        func1d(*arrs, *args, **kwargs)
    
    If there is only one array in `arrs` this is equivalent to::
    
        numpy.apply_along_axis(func1d, axis, arrs[0], *args, **kwargs)
        
    All arrays in `arrs` must have compatible dimensions to be able to run
    `numpy.concatenate(arrs, axis)`
    
    Arguments:
        func1d:   Function that operates on `len(arrs)` 1 dimensional arrays,
                  with signature `f(*arrs, *args, **kwargs)`
        axis:     Axis of all `arrs` to apply the function along
        arrs:     Iterable of numpy arrays
        *args:    Passed to func1d after array arguments
        **kwargs: Passed to func1d as keyword arguments
    """
    # Concatenate the input arrays along the calculation axis to make one big
    # array that can be passed in to `apply_along_axis`
    carrs = np.concatenate(arrs, axis)
    
    # We'll need to split the concatenated arrays up before we apply `func1d`,
    # here's the offsets to split them back into the originals
    offsets=[]
    start=0
    for i in range(len(arrs)-1):
        start += arrs[i].shape[axis]
        offsets.append(start)
            
    # The helper closure splits up the concatenated array back into the components of `arrs`
    # and then runs `func1d` on them
    def helperfunc(a, *args, **kwargs):
        arrs = np.split(a, offsets)
        return func1d(*[*arrs, *args], **kwargs)
    
    # Run `apply_along_axis` along the concatenated array
    return np.apply_along_axis(helperfunc, axis, carrs, *args, **kwargs)

#############################################################
#############################################################
#############################################################
#################### user inputs 

dtype = 'float64'
wood_dtype = 'uint8'
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

## start client
client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

# Create variable used for time axis
time_var = xr.Variable('time',times)

#############################################################
#########################################################

######### get regions 
regions = sorted(glob('../raw_data/GIS/MR*ID*.geojson'))
regions = [r for r in regions if 'pts' not in r]
print("{} regions".format(len(regions)))

geometries = []
for r in regions:
    with open(r) as f:
        gj = json.load(f)
    features = gj['features'][0]

    geometries.append(features['geometry'])


# fpoints = sorted(glob('../raw_data/GIS/MR_allpts_clipped_active_10m_wgs84.geojson'))
fpoints = sorted(glob('../raw_data/GIS/MR_allpts_clipped_active_5m_c_wgs84.geojson'))
with open(fpoints[0]) as f:
    gj = json.load(f)
features = gj['features']

points = [f['geometry']['coordinates'][0] for f in features]
print("{} sample points".format(len(points)))

#############################################################
#########################################################
## time-series at every point
veg_files = sorted(glob('../raw_data/MR/MR_veg/MR_*_Prob1_regrid.tif'))
water_files = sorted(glob('../raw_data/MR/MR_water/MR_*_Prob0_regrid.tif'))
dem_files = sorted(glob('../raw_data/Elwha_PlaneCamLidarDEMs_2013to2016/*MR_*DEM_regrid.tif'))
print(len(dem_files))

# get filtered wood probs, clipped to margins
# wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/Elwha_MR_*bin0.25_regrid_cc.tif'))
wood_files = sorted(glob('../results/MR/MR_wood/wood_detect/Elwha_MR_*bin0.1_regrid_cc.tif'))
print(len(wood_files))

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=wood_dtype) for i in wood_files], #dtype
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
wood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

# # get timeaverage image for consistent lighting
# avim_ds = rioxarray.open_rasterio("MR/MR_orthos_orig/Elwha_MR_im_time_mean_prob.tif", chunks=chunksize, dtype='uint8')
# avim_ds = avim_ds.to_dataset('band')
# print(avim_ds.dims)

#############################################################
# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in water_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
water_geotiffs_ds = geotiffs_ds.rename({1: 'water'})

#############################################################
# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in veg_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
veg_geotiffs_ds = geotiffs_ds.rename({1: 'veg'})

#############################################################

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in dem_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
dem_geotiffs_ds = geotiffs_ds.rename({1: 'dem'})

#########################################################
## clean up
water_geotiffs_ds = water_geotiffs_ds.drop_vars(2)
veg_geotiffs_ds = veg_geotiffs_ds.drop_vars(2)
wood_geotiffs_ds = wood_geotiffs_ds.drop_vars(2)
dem_geotiffs_ds = dem_geotiffs_ds.drop_vars(2)

print(water_geotiffs_ds.to_array().shape)
print(veg_geotiffs_ds.to_array().shape)
print(wood_geotiffs_ds.to_array().shape)
print(dem_geotiffs_ds.to_array().shape)

wood_geotiffs_ds = wood_geotiffs_ds.astype(wood_dtype) #dtype

# elev_bins = [0,2,4,8,10,12,]






#############################################################
#########################################################

x=np.array(points)[:,0]
y=np.array(points)[:,1]

dat_wood = np.zeros((len(x),len(times)))
dat_water = np.zeros((len(x),len(times)))
dat_veg = np.zeros((len(x),len(times)))
dat_dem = np.zeros((len(x),len(times)))

for counter, (xx,yy) in tqdm(enumerate(zip(x,y))):
    pwood = wood_geotiffs_ds.wood.sel(x=xx,y=yy, method="nearest")
    pwater = water_geotiffs_ds.water.sel(x=xx,y=yy, method="nearest")
    pveg = veg_geotiffs_ds.veg.sel(x=xx,y=yy, method="nearest")
    pdem = dem_geotiffs_ds.dem.sel(x=xx,y=yy, method="nearest")

    dat_wood[counter,:] = pwood.to_numpy()
    dat_water[counter,:] = pwater.to_numpy()
    dat_veg[counter,:] = pveg.to_numpy()
    dat_dem[counter,:] = pdem.to_numpy()

np.savez('../results/MR/MR_wood/summary/bin_wood_water_veg_dem_allpts_5m.npz', dat_veg=dat_veg, dat_water=dat_water, dat_wood=dat_wood, dat_dem=dat_dem, x=x, y=y)

# np.savez('../results/MR/bin_wood_water_veg_allpts.npz', dat_veg=dat_veg, dat_water=dat_water, dat_wood=dat_wood, x=x, y=y)
# np.savez('../results/MR/probs_wood_water_veg_allpts.npz', dat_veg=dat_veg, dat_water=dat_water, dat_wood=dat_wood, x=x, y=y)

# plt.scatter(x,y,10,np.mean(dat_wood,axis=1)); plt.show()
# plt.scatter(x,y,10,np.mean(dat_veg,axis=1)); plt.show()
# plt.scatter(x,y,10,np.mean(dat_water,axis=1)); plt.show()
# plt.scatter(x,y,10,np.mean(dat_dem,axis=1)); plt.show()

#########################################################

with np.load('../results/MR/MR_wood/summary/bin_wood_water_veg_dem_allpts_5m.npz') as f:
    dat_veg = f['dat_veg']
    dat_water = f['dat_water']
    dat_wood = f['dat_wood']
    dat_dem = f['dat_dem']    
    dat_x = f['x']
    dat_y = f['y']

dt = [datetime.strptime(time,'%Y-%m-%d') for time in times]

# plt.figure(figsize=(6,12))
# plt.subplot(311)
# plt.plot(dt,np.sum(dat_wood>0,0),'r')
# plt.subplot(312)
# plt.plot(dt,np.sum(dat_water>0,0),'b')
# plt.subplot(313)
# plt.plot(dt,np.sum(dat_veg>0,0),'g')
# plt.show()

#########################################################
#########################################################

plt.figure(figsize=(32,16))

for k in range(len(times)):
    tmp = dat_wood[:,k].copy()
    xtmp = x.copy()
    ytmp = y.copy()
    xtmp = xtmp[tmp>0.25]
    ytmp = ytmp[tmp>0.25]
    tmp = tmp[tmp>0.25]

    plt.subplot(1,14,k+1)
    plt.scatter(xtmp,ytmp,10,tmp)
    plt.axis('off')
    plt.title(times[k])

# plt.show()
plt.savefig("../results/MR/MR_wood/summary/Wood_bin_5m_sample_history.png", dpi=300, bbox_inches='tight')
plt.close()

tots = []
for k in range(len(times)):
    tmp = dat_wood[:,k].copy()
    xtmp = x.copy()
    ytmp = y.copy()
    xtmp = xtmp[tmp>0.25]
    ytmp = ytmp[tmp>0.25]
    tmp = tmp[tmp>0.25]
    tots.append(100*np.sum(tmp>0)/len(x))

plt.figure(figsize=(32,4))
plt.plot(dt,tots,'k-',marker='o',markerfacecolor='r', markeredgecolor='w')
plt.xlabel('Time')
plt.ylabel('$\%$ sample points\n occupied by wood')
# plt.show()
plt.savefig("../results/MR/MR_wood/summary/Percent_reach_wood_bin_sample_history_5m.png", dpi=300)
plt.close()



















#############################################################
#########################################################
## correlations over time, each region of filtered wood

# da_corr = xr.corr(wood_geotiffs_ds.wood.sel(time=times[0]), wood_geotiffs_ds.wood.sel(time=times[1]))
# da_corr.rio.to_raster(raster_path=f"../results/MR/MR_wood/Correl_wood_t0_t1.tif", dtype=dtype)

# autocorr = multi_apply_along_axis(pearsonr, 0, [wood_geotiffs_ds.wood, wood_geotiffs_ds.wood]) #.sel(time=times[1])

# from scipy.signal import correlate2d

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


### https://stackoverflow.com/questions/4503325/autocorrelation-of-a-multidimensional-array-in-numpy
from itertools import product

def autocorrelate(x):
    """
    Compute the multidimensional autocorrelation of an nd array.
    input: an nd array of floats
    output: an nd array of autocorrelations
    """

    # used for transposes
    t = np.roll(range(x.ndim), 1)

    # pairs of indexes
    # the first is for the autocorrelation array
    # the second is the shift
    ii = [list(enumerate(range(1, s - 1))) for s in x.shape]

    # initialize the resulting autocorrelation array
    acor = np.empty(shape=[len(s0) for s0 in ii])

    # iterate over all combinations of directional shifts
    for i in product(*ii):
        # extract the indexes for
        # the autocorrelation array 
        # and original array respectively
        i1, i2 = np.asarray(i).T

        x1 = x.copy()
        x2 = x.copy()

        for i0 in i2:
            # clip the unshifted array at the end
            x1 = x1[:-i0]
            # and the shifted array at the beginning
            x2 = x2[i0:]

            # prepare to do the same for 
            # the next axis
            x1 = x1.transpose(t)
            x2 = x2.transpose(t)

        # normalize shifted and unshifted arrays
        x1 -= x1.mean()
        x1 /= x1.std()
        x2 -= x2.mean()
        x2 /= x2.std()

        # compute the autocorrelation directly
        # from the definition
        acor[tuple(i1)] = (x1 * x2).mean()

    return acor


for counter,g in tqdm(enumerate(geometries)):

    wood_c = wood_geotiffs_ds.rio.clip([g], wood_geotiffs_ds.rio.crs)
    # autocorr = multi_apply_along_axis(pearsonr, 0, [wood_c.wood, wood_c.wood])

    # tmp1 = wood_c.sel(time=slice(times[0], times[1], times[2])).to_array()
    # tmp2 = wood_c.sel(time=slice(times[3], times[4], times[5])).to_array()
    # tmp0 = wood_c.wood.sel(time=slice(times[0])).to_numpy()
    # tmp1 = wood_c.wood.sel(time=slice(times[0])).to_numpy()
    # autocorr = multi_apply_along_axis(pearsonr, 0, [tmp1,tmp2]) 

    # xc = correlate2d(tmp0.squeeze(), tmp1.squeeze(), mode='valid')
    # xc = corr2_coeff(tmp0.squeeze(), tmp1.squeeze())
    # xc = generate_correlation_map(tmp0.squeeze(), tmp1.squeeze())

    xc = autocorrelate(wood_c.wood.to_numpy())

    # xc = multi_apply_along_axis(autocorrelate, 0, wood_c.wood)






# autocorr.shape

# fig, axes = plt.subplots(1,2, figsize=(10,3))

# p0 = axes[0].pcolormesh(corr[0,:,:])
# plt.colorbar(p0, ax=axes[0])
# axes[0].set_title('Pearson Correlation Coefficient')

# p1 = axes[1].pcolormesh(numpy.log(corr[1,:,:]))
# axes[1].set_title('Log p-value')
# plt.colorbar(p1, ax=axes[1])



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

# da = xr.tutorial.open_dataset('air_temperature')["air"].load()
# da_2013 = da.sel(time="2013")
# da_2014 = da.sel(time="2014")
# da_2013["time"] = da_2013["time"].dt.strftime("%m%d%H")
# da_2014["time"] = da_2014["time"].dt.strftime("%m%d%H")
# da_corr = xr.corr(da_2013, da_2014, dim="time")

# import bottleneck

# def covariance_gufunc(x, y):
#     return (
#         (x - x.mean(axis=-1, keepdims=True)) * (y - y.mean(axis=-1, keepdims=True))
#     ).mean(axis=-1)


# def pearson_correlation_gufunc(x, y):
#     return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))


# def spearman_correlation_gufunc(x, y):
#     x_ranks = bottleneck.rankdata(x, axis=-1)
#     y_ranks = bottleneck.rankdata(y, axis=-1)
#     return pearson_correlation_gufunc(x_ranks, y_ranks)


# def spearman_correlation(x, y, dim):
#     return xr.apply_ufunc(
#         spearman_correlation_gufunc,
#         x,
#         y,
#         input_core_dims=[[dim], [dim]],
#         dask="parallelized",
#         output_dtypes=[float],
#     )

# r = spearman_correlation(array1, array2, "time").compute()


#############################################################
#########################################################
## bin by elevation


#############################################################
#########################################################
## active versus non-active wood









# plt.figure(figsize=(10,6))
# pwood.plot(color=[.588, .294, 0.], label='wood'); pwater.plot(color='b', label='water')
# pveg.plot(color='g', label='veg'); pdev.plot(color=[.5,.5,.5], label='dev'); 
# plt.legend()
# plt.xlabel('Prediction probability')
# plt.xticks(rotation=45)
# plt.show()


# source_crs = wood_geotiffs_ds.rio.crs.data['init'] # Coordinate system of the file
# target_crs = 'epsg:4326' # Global lat-lon coordinate system

# polar_to_latlon = Transformer.from_crs(target_crs,source_crs)
# lat, lon = polar_to_latlon.transform(x, y, direction='inverse')
# print(lat)
# print(lon)

# xx=-123.60021506
# yy=48.00648881



## alternative method
# dat_wood = np.zeros((len(x),len(times)))
# dat_water = np.zeros((len(x),len(times)))
# dat_veg = np.zeros((len(x),len(times)))

# for counter, (xx,yy) in tqdm(enumerate(zip(x,y))):
#     for inner_counter, time in enumerate(times):
#         pwood = wood_geotiffs_ds.wood.sel(x=xx,y=yy, method="nearest").sel(time=time)
#         pwater = water_geotiffs_ds.water.sel(x=xx,y=yy, method="nearest").sel(time=time)
#         pveg = veg_geotiffs_ds.veg.sel(x=xx,y=yy, method="nearest").sel(time=time)
#         dat_wood[counter,inner_counter] = pwood
#         dat_water[counter,inner_counter] = pwater
#         dat_veg[counter,inner_counter] = pveg

# plt.figure(figsize=(10,6))
# pwood.plot(color=[.588, .294, 0.], label='wood'); pwater.plot(color='b', label='water')
# pveg.plot(color='g', label='veg'); pdev.plot(color=[.5,.5,.5], label='dev'); 
# plt.legend()
# plt.xlabel('Prediction probability')
# plt.xticks(rotation=45)
# plt.show()
