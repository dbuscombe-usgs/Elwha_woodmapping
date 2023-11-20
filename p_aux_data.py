

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


x = [1,3,5,7,9.5,13.5,19,27,38,53,77,107,154,216,256]

Dyears=[]
Syears=[]
D=[]
for k in grainsize_files2001:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))

D=[]
for k in grainsize_files2002:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))

D=[]
for k in grainsize_files2003:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))

D=[]
for k in grainsize_files2004:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))

D=[]
for k in grainsize_files2005:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))

D=[]
for k in grainsize_files2006:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))

D=[]
for k in grainsize_files2010:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))

D=[]
for k in grainsize_files2014:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))

D=[]
for k in grainsize_files2015:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))

D=[]
for k in grainsize_files2016:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))

D=[]
for k in grainsize_files2021:
    dat = pd.read_csv(k)
    y = dat['Cum %']
    d = np.interp(50,y,x)
    D.append(d)
Dyears.append(np.mean(D))
Syears.append(np.std(D))


rack_vol = pd.read_csv(rack_vol_file[0],header=None)
rack_years = [2013,2014,2015,2016,2018]

dep_ft = pd.read_csv(pool_depth_file[0],header=None)
pool_years = [2000, 2002, 2003, 2004, 2006, 2010, 2012, 2013, 2014, 2015, 2016, 2018, 2021]


########################################
plt.figure(figsize=(8,8))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(221)
plt.plot(times, [len(f) for f in all_dirs], 'k-o')
plt.ylabel('Number of LR bars \n sampled for grain size')
plt.title('a)', loc='left')

plt.gca().set_xticks(times)
plt.gca().set_xticklabels([str(t) for t in times], rotation = 45)
plt.ylim(0,8)

plt.subplot(222)
plt.errorbar(times, Dyears, Syears, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
plt.plot(times, Dyears,'--',color=[.5,.5,.5])
plt.ylabel('Median bulk grain size (mm)')
plt.title('b)', loc='left')

plt.gca().set_xticks(times)
plt.gca().set_xticklabels([str(t) for t in times], rotation = 45)
plt.ylim(0,125)

plt.subplot(223)
plt.plot(rack_years, np.nanmean(rack_vol, axis=1), 'k-o')
plt.ylabel(r'Mean wood rack volume (m$^3$)')
plt.title('c)', loc='left')
plt.ylim(0,900)

plt.subplot(224)
plt.plot(pool_years, np.nanmean(dep_ft, axis=1)*0.3048, 'k-o')
plt.ylabel(r'Mean pool depth (m)')
plt.title('d)', loc='left')
plt.ylim(0,2.1)

# plt.show()
plt.savefig("summaries/LR_wood_grainsize_measurements.png", dpi=300, bbox_inches="tight")
plt.close()


