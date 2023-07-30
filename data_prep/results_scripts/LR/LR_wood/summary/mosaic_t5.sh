rm mosaic.vrt

gdalbuildvrt -input_file_list alltifs_t5.txt mosaic.vrt
rm Elwha_LR_2014-02-01_wood_filtered_prob.tif
gdal_translate -co "COMPRESS=LZW" mosaic.vrt Elwha_LR_2014-02-01_wood_filtered_prob.tif
rm Elwha_LR_2014-02-01_wood_filtered_bin0.1_regrid.tif

rm Elwha_LR_2014-02-01_wood_filtered_bin0.1.tif
gdal_calc.py -A Elwha_LR_2014-02-01_wood_filtered_prob.tif --outfile=Elwha_LR_2014-02-01_wood_filtered_bin0.1.tif --calc="A>.1" --NoDataValue=0

rm Elwha_LR_2014-02-01_wood_filtered_bin0.1_regrid.tif
gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 Elwha_LR_2014-02-01_wood_filtered_bin0.1.tif Elwha_LR_2014-02-01_wood_filtered_bin0.1_regrid.tif 

rm Elwha_LR_2014-02-01_wood_filtered_bin0.1.tif 