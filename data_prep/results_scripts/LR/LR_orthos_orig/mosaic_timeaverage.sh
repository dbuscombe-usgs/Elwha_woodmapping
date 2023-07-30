rm mosaic.vrt
gdalbuildvrt -input_file_list alltifs.txt mosaic.vrt
rm Elwha_LR_im_time_mean_prob.tif
gdal_translate -co "COMPRESS=LZW" mosaic.vrt Elwha_LR_im_time_mean_prob.tif

rm Elwha_LR_im_time_bin0.tif
gdal_calc.py -A Elwha_LR_im_time_mean_prob.tif --outfile=Elwha_LR_im_time_bin0.tif --calc="A>0" --NoDataValue=0

rm Elwha_LR_im_time_bin0_regrid.tif
gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 Elwha_LR_im_time_bin0.tif Elwha_LR_im_time_bin0_regrid.tif

rm Elwha_LR_im_time_bin0.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 Elwha_LR_im_time_mean_prob.tif Elwha_LR_im_time_mean_prob_regrid.tif