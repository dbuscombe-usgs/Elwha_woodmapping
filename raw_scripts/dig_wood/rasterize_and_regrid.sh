

gdal_rasterize -l LR_20170922_wood_dig_clipped_active_budgetextent_epsg6339 -burn 1.0 -tr 0.125 0.125 -a_nodata 0.0 -te 457704.9227 5326631.3483 459241.6735 5333311.0001 -ot Float32 -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 LR_20170922_wood_dig_clipped_active_budgetextent_epsg6339.geojson LR_20170922_dig_wood_clipped_active_budgetextent.tif

gdal_rasterize -l LR_20120407_wood_dig_clipped_active_budgetextent_epsg6339 -burn 1.0 -tr 0.125 0.125 -a_nodata 0.0 -te 457704.9227 5326631.3483 459241.6735 5333311.0001 -ot Float32 -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 LR_20120407_wood_dig_clipped_active_budgetextent_epsg6339.geojson LR_20120407_dig_wood_clipped_active_budgetextent.tif

gdal_rasterize -l MR_20120407_wood_dig_clipped_active_budgetextent_epsg6339 -burn 1.0 -tr 0.125 0.125 -a_nodata 0.0 -te 455157.2495 5316533.0111 457076.1213 5323771.7304 -ot Float32 -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 MR_20120407_wood_dig_clipped_active_budgetextent_epsg6339.geojson MR_20120407_dig_wood_clipped_active_budgetextent.tif

gdal_rasterize -l MR_20170922_wood_dig_clipped_active_budgetextent_epsg6339 -burn 1.0 -tr 0.125 0.125 -a_nodata 0.0 -te 455157.2495 5316533.0111 457076.1213 5323771.7304 -ot Float32 -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 MR_20170922_wood_dig_clipped_active_budgetextent_epsg6339.geojson MR_20170922_dig_wood_clipped_active_budgetextent.tif

# gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 LR_20170922_dig_wood_bin.tif LR_20170922_dig_wood_bin_c.tif

# gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 MR_20170922_dig_wood_bin.tif MR_20170922_dig_wood_bin_c.tif

# gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 MR_20120407_dig_wood_bin.tif MR_20120407_dig_wood_bin_c.tif