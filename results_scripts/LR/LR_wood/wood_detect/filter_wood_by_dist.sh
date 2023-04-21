#####
gdal_calc.py -A Elwha_LR_2012-04-07_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2012-04-07_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2012-04-07_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

####
gdal_calc.py -A Elwha_LR_2012-08-10_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2012-08-10_c_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2012-08-10_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2012-11-08_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2012-11-08_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2012-11-08_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2013-02-13_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2013-02-13_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2013-02-13_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2013-04-30_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2013-04-30_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2013-04-30_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2013-09-19_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2013-09-19_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2013-09-19_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2014-02-01_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2014-02-01_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2014-02-01_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2014-09-30_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2014-09-30_c_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2014-09-30_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2015-03-03_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2015-03-03_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2015-03-03_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2015-09-23_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2015-09-23_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2015-09-23_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2016-01-01_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2016-01-11_c_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2016-01-01_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2016-07-14_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2016-07-14_c_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2016-07-14_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2016-09-30_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2016-09-30_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2016-09-30_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_LR_2017-09-22_wood_filtered_bin0.1_regrid_cc.tif -B ../../LR_dist2braid/LR_dist_to_braid_2017-09-22_regrid.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_LR_2017-09-22_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif