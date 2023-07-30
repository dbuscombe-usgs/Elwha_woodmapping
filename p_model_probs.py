
## Dan Buscombe, Marda Science
## Apr, 2023
##

import json, os
import rioxarray
import xarray as xr 
from glob import glob 
from dask.distributed import Client
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import ndimage
from skimage.exposure import match_histograms
from matplotlib.colors import ListedColormap

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

# n_workers = 22
# threads_per_worker = 2
# memory_limit='115GB'
## start client
# client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

cwd = os.getcwd()


# Create variable used for time axis
time_var = xr.Variable('time',times)

######### get movie regions and clipper
movie_regions = sorted(glob('../raw_data/GIS/LR*movie*epsg6339.geojson'))
print("{} movie_ regions".format(len(movie_regions)))

movie_geometries = []
for r in movie_regions:
    with open(r) as f:
        gj = json.load(f)
    features = gj['features'][0]

    movie_geometries.append(features['geometry'])


r = "../raw_data/GIS/LR_movie_bars.geojson"
with open(r) as f:
    gj = json.load(f)
LR_bars = [a['geometry'] for a in gj['features']]


# fourclass_files = sorted(glob('../results/LR/LR_all/model_out/*_4classMosaic.tif'))

# #############################################################
# # Load in and concatenate all individual GeoTIFFs
# geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in fourclass_files],
#                         dim=time_var)
# # Covert our xarray.DataArray into a xarray.Dataset
# label_geotiffs_ds = geotiffs_da.to_dataset('band')


#############################################################
im_files = sorted(glob('../raw_data/LR/LR_orthos_orig/Elwha_LR_*_regrid.tif'))
im_files = [i for i in im_files if 'bin' not in i]
print(len(im_files))

im_files_filt = []
for t in times:
    t = t.replace('-','')
    tmp = [d for d in im_files if t in d]
    im_files_filt.append(tmp[0])

geotiffs_da = xr.concat([rioxarray.open_rasterio(i, chunks=chunksize, dtype=dtype) for i in im_files_filt],
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
reference = im_geotiffs_ds.sel(time='2016-07-14')
# reference = im_geotiffs_ds.sel(time='2015-03-03')


# cmap = ListedColormap(["black","lawngreen", "blue", "gold", "brown"])

# ## all - instantanues
# for counter,g in tqdm(enumerate(LR_bars)):
#     print("Working on region {}".format(counter))

#     ref_c = reference.rio.clip([g], avim_ds.rio.crs)
#     im_c = im_geotiffs_ds.rio.clip([g], avim_ds.rio.crs)
#     label_c = label_geotiffs_ds.rio.clip([g], label_geotiffs_ds.rio.crs)
#     tmp_da = xr.concat([im_c.red,im_c.green,im_c.blue],dim=('x','x','x'))
#     reftmp_da = xr.concat([ref_c.red,ref_c.green,ref_c.blue],dim=('x','x','x'))

#     for inner_counter, time in enumerate(times):
#         print("Working on time {}".format(time))

#         im_da = tmp_da.sel(time=time).transpose()/255.
#         refim_da = reftmp_da.transpose()/255.
#         matched = match_histograms(im_da.to_numpy(), refim_da.to_numpy(), channel_axis=-1)
#         label_da = label_c[1].sel(time=time)
#         label_da = label_da.transpose().to_numpy()

#         fig1, ax1 = plt.subplots()
#         ax1.imshow(matched)
#         ax1.imshow(label_da, cmap=cmap, alpha=0.3)
#         plt.title(time)
#         plt.axis('off')

#         # plt.show()
#         plt.savefig(f"../results/LR/LR_Bars_All_inst_movie_{counter}_time_{time}.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         del label_da


