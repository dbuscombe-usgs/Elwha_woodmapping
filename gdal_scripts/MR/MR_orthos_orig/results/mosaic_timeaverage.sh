gdalbuildvrt -input_file_list alltifs.txt mosaic.vrt
gdal_translate -co "COMPRESS=LZW" mosaic.vrt Elwha_MR_im_time_mean_prob.tif

gdal_calc.py -A Elwha_MR_im_time_mean_prob.tif --outfile=Elwha_MR_im_time_bin0.tif --calc="A>0" --NoDataValue=0

gdalwarp -cutline grid.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr 1.569605128802169152e-06 -1.569621352017918482e-06 Elwha_MR_im_time_bin0.tif Elwha_MR_im_time_bin0_regrid.tif

rm Elwha_MR_im_time_bin0.tif