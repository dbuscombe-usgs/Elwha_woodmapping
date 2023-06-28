## Dan Buscombe, Marda Science
## Apr-June, 2023
#

import json, os
import rioxarray
import xarray as xr 
from glob import glob 
import matplotlib.pyplot as plt
import numpy as np
# from dask.distributed import Client
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from area import area
from skimage.measure import label, regionprops_table


#############################################################
#############################################################
#############################################################
#################### user inputs 

dtype = 'float64'
chunksize = ("auto", "auto")

times = [
    '2012-04-07',
    '2012-08-10',
    '2012-11-08',
    '2013-02-13',
    '2013-04-30',
    '2013-09-19',
    '2014-02-01',
    '2014-09-30',
    '2015-03-03',
    '2015-09-23',
    '2016-01-11',
    '2016-07-14',
    '2016-09-30',
    '2017-09-22'
]


elj_file = '../raw_data/20101208_ELJ_EX/20101208_ELJ_EX.geojson'
with open(elj_file) as f:
    gj = json.load(f)
ELJs = gj['features']

years = [f['properties']['YEAR_BUILT'] for f in ELJs]

