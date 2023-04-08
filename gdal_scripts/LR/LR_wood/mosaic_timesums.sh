gdalbuildvrt -input_file_list allsumtifs.txt mosaic.vrt
gdal_translate -co "COMPRESS=LZW" mosaic.vrt Elwha_LR_wood_filtered_time_sum.tif

gdalwarp -cutline LRgrid.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr 1.569605128802169152e-06 -1.569621352017918482e-06 Elwha_LR_wood_filtered_time_sum.tif Elwha_LR_wood_filtered_time_sum_regrid.tif
