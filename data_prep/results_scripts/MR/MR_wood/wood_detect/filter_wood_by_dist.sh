#####
gdal_calc.py -A Elwha_MR_2012-04-07_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2012-04-07.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2012-04-07_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

####
gdal_calc.py -A Elwha_MR_2012-08-10_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2012-08-10_c.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2012-08-10_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2012-11-08_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2012-11-08.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2012-11-08_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2013-02-13_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2013-02-13.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2013-02-13_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2013-04-30_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2013-04-30.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2013-04-30_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2013-09-19_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2013-09-19_c.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2013-09-19_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2014-02-01_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2014-02-01.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2014-02-01_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2014-09-30_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2014-09-30_c.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2014-09-30_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2015-03-03_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2015-03-03.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2015-03-03_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2015-09-23_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2015-09-23.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2015-09-23_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2016-01-01_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2016-01-11_c.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2016-01-01_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2016-07-14_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2016-07-14_c.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2016-07-14_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2016-09-30_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2016-09-30.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2016-09-30_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif

#####
gdal_calc.py -A Elwha_MR_2017-09-22_wood_filtered_bin0.1_regrid_cc.tif -B ../../MR_dist2braid/MR_dist_to_braid_2017-09-22.tif --outfile=result.tif --calc="A*(B<500)"

gdalwarp -cutline grid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2017-09-22_wood_filtered_bin0.1_regrid_ccc.tif

rm result.tif