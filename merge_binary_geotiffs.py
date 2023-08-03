

import rioxarray
import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt

outfile = "label2023.tif"
woodfile = 'wood2023.tif'
sandfile = 'sand2023.tif'
mixedfile = 'mixed2023.tif'
coarsefile = 'coarse2023.tif'
rockfile = 'rock2023.tif'
waterfile = 'water2023.tif'

wood = rioxarray.open_rasterio(woodfile)

sand = rioxarray.open_rasterio(sandfile)

mixed = rioxarray.open_rasterio(mixedfile)

coarse = rioxarray.open_rasterio(coarsefile)

rock = rioxarray.open_rasterio(rockfile)

water = rioxarray.open_rasterio(waterfile)

stack = np.dstack((np.zeros_like(water.squeeze()), water.squeeze(),sand.squeeze(),mixed.squeeze(),coarse.squeeze(),rock.squeeze(),wood.squeeze()))

label = np.argmax(stack,axis=-1)

plt.imshow(label); plt.colorbar(); plt.show()

y = water['y'].values
x = water['x'].values

output = xr.DataArray(label, coords={'y':y, 'x':x}, dims=['y', 'x'])

output.rio.to_raster(outfile, compress='LZW', tiled=True, dtype="int8")