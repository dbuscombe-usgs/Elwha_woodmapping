gdalbuildvrt -input_file_list alltifs_t11.txt mosaic.vrt
gdal_translate -co "COMPRESS=LZW" mosaic.vrt Elwha_MR_2016-09-30_wood_filtered_prob.tif

gdal_calc.py -A Elwha_MR_2016-09-30_wood_filtered_prob.tif --outfile=Elwha_MR_2016-09-30_wood_filtered_bin0.25.tif --calc="A>.25" --NoDataValue=0

gdalwarp -cutline grid.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr 1.569605128802169152e-06 -1.569621352017918482e-06 Elwha_MR_2016-09-30_wood_filtered_bin0.25.tif Elwha_MR_2016-09-30_wood_filtered_bin0.25_regrid.tif 

rm Elwha_MR_2016-09-30_wood_filtered_bin0.25.tif 