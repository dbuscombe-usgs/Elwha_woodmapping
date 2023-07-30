# rm mosaic.vrt
# gdalbuildvrt -input_file_list alltifs.txt mosaic.vrt
# rm Elwha_LR_sed_time_mean_prob.tif
# gdal_translate -co "COMPRESS=LZW" mosaic.vrt Elwha_LR_sed_time_mean_prob.tif

rm Elwha_LR_sed_time_bin0.9.tif
gdal_calc.py -A Elwha_LR_sed_time_mean_prob.tif --outfile=Elwha_LR_sed_time_bin0.9.tif --calc="A>.9" --NoDataValue=0

rm Elwha_LR_sed_time_bin0.9_regrid.tif
gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 Elwha_LR_sed_time_bin0.9.tif Elwha_LR_sed_time_bin0.9_regrid.tif

rm Elwha_LR_sed_time_bin0.9.tif