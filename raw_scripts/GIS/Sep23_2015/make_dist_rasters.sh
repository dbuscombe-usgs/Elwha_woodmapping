
#### LR
ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:4326 -s_srs EPSG:32610 Elwha_braids_LR_20150923_reproj.shp Elwha_braids_LR_20150923.shp

gdal_rasterize -burn 1.0 -tr 1.5696051288021692e-06 1.5696051288021692e-06 -a_nodata 0.0 -te -123.568 48.091643668 -123.548 48.151643668 -ot Byte -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 Elwha_braids_LR_20150923_reproj.shp LR_braid_2015-09-23.tif

gdal_proximity.py -srcband 1 -distunits PIXEL -nodata 0.0 -ot Float32 -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 LR_braid_2015-09-23.tif LR_dist_to_braid_2015-09-23.tif

#### MR
ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:4326 -s_srs EPSG:32610 Elwha_braids_MR_20150923_reproj.shp Elwha_braids_MR_20150923.shp

gdal_rasterize -burn 1.0 -tr 1.5696051288021692e-06 1.5696051288021692e-06 -a_nodata 0.0 -te -123.6011542 48.0006500 -123.5761535 48.0656496 -ot Byte -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 Elwha_braids_MR_20150923_reproj.shp MR_braid_2015-09-23.tif

gdal_proximity.py -srcband 1 -distunits PIXEL -nodata 0.0 -ot Float32 -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 MR_braid_2015-09-23.tif MR_dist_to_braid_2015-09-23.tif
