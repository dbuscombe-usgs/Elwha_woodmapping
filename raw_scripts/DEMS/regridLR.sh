gdalwarp -t_srs EPSG:6339 2013_PlaneCamLidar_Final.tif 2013_PlaneCamLidar_Final_6339.tif
gdalwarp -t_srs EPSG:6339 2014_PlaneCamLidar_Final.tif 2014_PlaneCamLidar_Final_6339.tif
gdalwarp -t_srs EPSG:6339 2015_PlaneCamLidar_Final.tif 2015_PlaneCamLidar_Final_6339.tif
gdalwarp -t_srs EPSG:6339 2016_PlaneCamLidar_Final.tif 2016_PlaneCamLidar_Final_6339.tif

gdalwarp  -cutline ../GIS/Apr07_2012/Elwha_margin_poly_LR_20120407.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2013_PlaneCamLidar_Final_6339.tif Elwha_LR_20120407_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20120407_DEM_regrid_cm.tif Elwha_LR_20120407_DEM_regrid.tif

gdalwarp -cutline ../GIS/Nov08_2012/Elwha_margin_poly_LR_20121108.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2013_PlaneCamLidar_Final_6339.tif Elwha_LR_20121108_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20121108_DEM_regrid_cm.tif Elwha_LR_20121108_DEM_regrid.tif

gdalwarp -cutline ../GIS/Nov08_2012/Elwha_margin_poly_LR_20121108.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2013_PlaneCamLidar_Final_6339.tif Elwha_LR_20120810_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20120810_DEM_regrid_cm.tif Elwha_LR_20120810_DEM_regrid.tif

gdalwarp -cutline ../GIS/Feb13_2013/Elwha_margin_poly_LR_20130213.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2013_PlaneCamLidar_Final_6339.tif Elwha_LR_20130213_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20130213_DEM_regrid_cm.tif Elwha_LR_20130213_DEM_regrid.tif

gdalwarp -cutline ../GIS/Apr30_2013/Elwha_margin_poly_LR_20130430.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2013_PlaneCamLidar_Final_6339.tif Elwha_LR_20130430_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20130430_DEM_regrid_cm.tif Elwha_LR_20130430_DEM_regrid.tif

gdalwarp -cutline ../GIS/Apr30_2013/Elwha_margin_poly_LR_20130430.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2013_PlaneCamLidar_Final_6339.tif Elwha_LR_20130919_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20130919_DEM_regrid_cm.tif Elwha_LR_20130919_DEM_regrid.tif

gdalwarp -cutline ../GIS/Feb01_2014/Elwha_margin_poly_LR_20140201.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2014_PlaneCamLidar_Final_6339.tif Elwha_LR_20140201_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20140201_DEM_regrid_cm.tif Elwha_LR_20140201_DEM_regrid.tif

gdalwarp -cutline ../GIS/Mar03_2015/Elwha_margin_poly_LR_20150303.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2014_PlaneCamLidar_Final_6339.tif Elwha_LR_20140930_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20140930_DEM_regrid_cm.tif Elwha_LR_20140930_DEM_regrid.tif

gdalwarp -cutline ../GIS/Mar03_2015/Elwha_margin_poly_LR_20150303.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2015_PlaneCamLidar_Final_6339.tif Elwha_LR_20150303_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20150303_DEM_regrid_cm.tif Elwha_LR_20150303_DEM_regrid.tif

gdalwarp -cutline ../GIS/Sep23_2015/Elwha_margin_poly_LR_20150923.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2015_PlaneCamLidar_Final_6339.tif Elwha_LR_20150930_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20150930_DEM_regrid_cm.tif Elwha_LR_20150930_DEM_regrid.tif

gdalwarp -cutline ../GIS/Sep23_2015/Elwha_margin_poly_LR_20150923.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2016_PlaneCamLidar_Final_6339.tif Elwha_LR_20160111_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20160111_DEM_regrid_cm.tif Elwha_LR_20160111_DEM_regrid.tif

gdalwarp -cutline ../GIS/Sep30_2016/Elwha_margin_poly_LR_20160930.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2016_PlaneCamLidar_Final_6339.tif Elwha_LR_20160714_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20160714_DEM_regrid_cm.tif Elwha_LR_20160714_DEM_regrid.tif

gdalwarp -cutline ../GIS/Sep30_2016/Elwha_margin_poly_LR_20160930.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2016_PlaneCamLidar_Final_6339.tif Elwha_LR_20160930_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20160930_DEM_regrid_cm.tif Elwha_LR_20160930_DEM_regrid.tif

gdalwarp -cutline ../GIS/Sep22_2017/Elwha_margin_poly_LR_20170922.shp -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 -t_srs EPSG:6339 2016_PlaneCamLidar_Final_6339.tif Elwha_LR_20170922_DEM_regrid_cm.tif

gdalwarp -cutline LRgrid_epsg6339.geojson -crop_to_cutline -dstalpha -co "COMPRESS=LZW" -tr .125 .125 Elwha_LR_20170922_DEM_regrid_cm.tif Elwha_LR_20170922_DEM_regrid.tif

rm *LR*regrid_cm.tif
rm *_6339.tif

