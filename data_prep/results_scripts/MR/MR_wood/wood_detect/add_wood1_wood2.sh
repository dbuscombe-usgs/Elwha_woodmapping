

gdal_calc.py -A Elwha_MR_2012-04-07_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2012-04-07_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2012-04-07_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2012-08-10_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2012-08-10_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2012-08-10_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2012-11-08_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2012-11-08_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2012-11-08_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2013-02-13_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2013-02-13_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2013-02-13_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2013-04-30_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2013-04-30_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2013-04-30_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2013-09-19_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2013-09-19_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2013-09-19_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2014-02-01_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2014-02-01_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2014-02-01_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2014-09-30_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2014-09-30_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2014-09-30_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2015-03-03_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2015-03-03_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2015-03-03_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2015-09-23_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2015-09-23_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2015-09-23_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2016-01-01_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2016-01-01_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2016-01-01_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2016-07-14_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2016-07-14_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2016-07-14_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2016-09-30_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2016-09-30_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2016-09-30_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif

gdal_calc.py -A Elwha_MR_2017-09-22_wood_filtered_bin0.1_regrid_cc.tif -B Elwha_MR_2017-09-22_wood2_filtered_bin0.1_regrid_cc.tif --outfile=result.tif --calc="(A+B)>0"
gdalwarp -cutline MRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs epsg:6339 result.tif Elwha_MR_2017-09-22_wood_filtered_bin0.1_regrid_ccc.tif
rm result.tif
