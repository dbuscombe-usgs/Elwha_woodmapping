gdalbuildvrt -input_file_list alltifs.txt mosaic.vrt
gdal_translate -co "COMPRESS=LZW" mosaic.vrt Elwha_LR_veg_time_mean_prob.tif

gdal_calc.py -A Elwha_LR_veg_time_mean_prob.tif --outfile=Elwha_LR_veg_time_bin0.9.tif --calc="A>.9" --NoDataValue=0

gdalwarp -cutline LRgrid.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr 1.569605128802169152e-06 -1.569621352017918482e-06 Elwha_LR_veg_time_bin0.9.tif Elwha_LR_veg_time_bin0.9_regrid.tif

rm Elwha_LR_veg_time_bin0.9.tif