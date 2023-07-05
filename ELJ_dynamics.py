## Dan Buscombe, Marda Science
## Apr-June, 2023
#

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


elj_file = '../raw_data/20101208_ELJ_EX/20101208_ELJ_EX.geojson'
with open(elj_file) as f:
    gj = json.load(f)
ELJs = gj['features']

years = [f['properties']['YEAR_BUILT'] for f in ELJs]

years = np.array(years)
years[years<1] = 2004.0

x = [f['geometry']['coordinates'][0] for f in ELJs]
y = [f['geometry']['coordinates'][1] for f in ELJs]

elj_zone = '../raw_data/20101208_ELJ_EX/ELJ_zone.geojson'
with open(elj_zone) as f:
    gj = json.load(f)
elj_zone = gj['features'][0]['geometry']



## make a plot of ELJs color-coded by year built

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

dt = [datetime.strptime(time,'%Y-%m-%d') for time in times]

# Create variable used for time axis
time_var = xr.Variable('time',times)

# # get timeaverage image for consistent lighting
# avim_ds = rioxarray.open_rasterio("../results/LR/LR_orthos_orig/Elwha_LR_im_time_mean_prob_regrid.tif", chunks=chunksize, dtype='uint8')
# avim_ds = avim_ds.to_dataset('band')
# print(avim_ds.dims)
# avim_ds = avim_ds.rename({1: 'red'})
# avim_ds = avim_ds.rename({2: 'green'})
# avim_ds = avim_ds.rename({3: 'blue'})
# avim_ds = avim_ds.drop_vars(4)

# av_c = avim_ds.rio.clip([elj_zone], avim_ds.rio.crs)
# av_da = xr.concat([av_c.red,av_c.green,av_c.blue],dim=('x','x','x'))
# av_da = av_da.transpose()/255. #.transpose()

#############################################################
im_files = sorted(glob('../raw_data/LR/LR_orthos_orig/Elwha_LR_*_regrid_c.tif'))
im_files = [i for i in im_files if 'bin' not in i]
print(len(im_files))

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in im_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
im_geotiffs_ds = geotiffs_ds.rename({1: 'red'})
im_geotiffs_ds = im_geotiffs_ds.rename({2: 'green'})
im_geotiffs_ds = im_geotiffs_ds.rename({3: 'blue'})
im_geotiffs_ds = im_geotiffs_ds.drop_vars(4)
print(im_geotiffs_ds.to_array().shape)

### reference image (bright)
reference1 = im_geotiffs_ds.sel(time='2012-04-07')
ref_c1 = reference1.rio.clip([elj_zone], reference1.rio.crs)
reftmp_da1 = xr.concat([ref_c1.red,ref_c1.green,ref_c1.blue],dim=('x','x','x'))
refim_da1 = reftmp_da1.transpose()/255.

reference2 = im_geotiffs_ds.sel(time='2017-09-22')
ref_c2 = reference2.rio.clip([elj_zone], reference2.rio.crs)
reftmp_da2 = xr.concat([ref_c2.red,ref_c2.green,ref_c2.blue],dim=('x','x','x'))
refim_da2 = reftmp_da2.transpose()/255.


###### wood
wood_files = sorted(glob('../results/LR/LR_wood/wood_detect/model1/LR_*cleaned.tif'))
print(len(wood_files))
# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in wood_files],
                        dim=time_var)
# Covert our xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')
# Rename the variable to a more useful name
wood_geotiffs_ds = geotiffs_ds.rename({1: 'wood'})

wood_c = wood_geotiffs_ds.rio.clip([elj_zone], wood_geotiffs_ds.rio.crs)

wood_da = wood_c.wood.sum("time", skipna=True).to_numpy()
wood_da[wood_da==0] = np.nan
wood_da = wood_da.transpose()/len(times) #


#### gamma correct the 2017 dark imagery
from PIL import Image
im_1_22 = 255.0 * (refim_da2 / 255.0)**(1 / 2.2)

##========================================================
def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn


im_1_22 = rescale_array(im_1_22.to_numpy(),0,255)
# im_22 = 255.0 * (refim_da2 / 255.0)**2.2

ind = np.argsort(years)
xsorted = np.array(x)[ind]
ysorted = np.array(y)[ind]

tmp = []
for xx,yy in zip(xsorted,ysorted):
    tmp.append(wood_c.wood.sel(x=xx,y=yy, method="nearest").to_numpy())

tmp = np.vstack(tmp)
tmp[tmp==0]=np.nan





plt.figure(figsize=(16,16))
plt.subplots_adjust(wspace=0.2, hspace=0.2)

plt.subplot(221)
# refim_da.plot.imshow()
plt.imshow(np.rot90(refim_da1), origin="lower", extent = [refim_da1.x.min().to_numpy(),refim_da1.x.max().to_numpy(),refim_da1.y.min().to_numpy(),refim_da1.y.max().to_numpy()]) #aspect="equal"
plt.setp( plt.gca().xaxis.get_majorticklabels(), rotation=45 )

scatt = plt.scatter(x,y,20,years, cmap='Reds', lw=1, edgecolors='k')
cbar=plt.colorbar(scatt, shrink=0.5)
cbar.set_label("ELJ construction date")
plt.ylabel('Northing (m)')
plt.xlabel('Easting (m)')
plt.title("a)", loc='left')

plt.subplot(222)
plt.imshow(np.rot90(im_1_22.astype(np.uint8)), origin="lower", extent = [refim_da2.x.min().to_numpy(),refim_da2.x.max().to_numpy(),refim_da2.y.min().to_numpy(),refim_da2.y.max().to_numpy()]) #aspect="equal"
plt.setp( plt.gca().xaxis.get_majorticklabels(), rotation=45 )
plt.plot(x,y,'wx', markersize=4, label='ELJ')
# plt.legend()
im=plt.imshow(np.rot90(wood_da), alpha=1, cmap='inferno', vmin=0, vmax=1, origin='lower', extent = [refim_da2.x.min().to_numpy(),refim_da2.x.max().to_numpy(),refim_da2.y.min().to_numpy(),refim_da2.y.max().to_numpy()])

plt.setp( plt.gca().xaxis.get_majorticklabels(), rotation=45)

cbar = plt.colorbar(im, shrink=0.5)
cbar.set_label("Wood persistence (-)")
plt.ylabel('Northing (m)')
plt.xlabel('Easting (m)')
plt.title("b)", loc='left')

plt.subplot(223)
plt.pcolormesh(dt,np.arange(37), tmp, cmap='bwr')
plt.gca().set_yticks(np.arange(0,37,2))
plt.gca().set_yticklabels(years[ind][::2])
plt.ylabel('Year of ELJ construction')
plt.title("c)", loc='left')

plt.subplot(224)
plt.plot(dt,np.nansum(tmp,axis=0))
plt.ylabel('Number of active ELJs')
plt.title("d)", loc='left')

# plt.show()
plt.savefig("summaries/ELJ_dynamics_timesummery.png", dpi=300, bbox_inches="tight")
plt.close()