

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from glob import glob 

## pool depth
pool_depth_file = sorted(glob('../raw_data/tribe_data/pool_depths_ft.csv'))

## rack volume
rack_vol_file = sorted(glob('../raw_data/tribe_data/rack_volume_m3.csv'))

## grain size
grainsize_files2001 = sorted(glob('../raw_data/tribe_data/*2001.csv'))
grainsize_files2002 = sorted(glob('../raw_data/tribe_data/*2002.csv'))
grainsize_files2003 = sorted(glob('../raw_data/tribe_data/*2003.csv'))
grainsize_files2004 = sorted(glob('../raw_data/tribe_data/*2004.csv'))
grainsize_files2005 = sorted(glob('../raw_data/tribe_data/*2005.csv'))
grainsize_files2006 = sorted(glob('../raw_data/tribe_data/*2006.csv'))
grainsize_files2010 = sorted(glob('../raw_data/tribe_data/*2010.csv'))
grainsize_files2014 = sorted(glob('../raw_data/tribe_data/*2014.csv'))
grainsize_files2015 = sorted(glob('../raw_data/tribe_data/*2015.csv'))
grainsize_files2016 = sorted(glob('../raw_data/tribe_data/*2016.csv'))
grainsize_files2021 = sorted(glob('../raw_data/tribe_data/*2021.csv'))

times = [2001,2002,2003,2004,2005,2006,2010,2014,2015,2016,2021]

all_dirs = [grainsize_files2001, grainsize_files2002, grainsize_files2003, grainsize_files2004, grainsize_files2005, grainsize_files2006, grainsize_files2010, grainsize_files2014, grainsize_files2015, grainsize_files2016,grainsize_files2021]

plt.plot(times, [len(f) for f in all_dirs])

