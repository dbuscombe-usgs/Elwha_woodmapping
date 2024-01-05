## Dan Buscombe, Marda Science
## 2023
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import mchmm as mc
import seaborn as sns
from matplotlib.colors import ListedColormap


###############################
########### analysis of transition


# ## veg, water, sed, wood
# PM = []; O2M = []; O3M = []
# ## all - instantanues
# for counter,g in tqdm(enumerate(MR_bars[7:])):
#     print("Working on region {}".format(7+counter))
#     label_c = label_geotiffs_ds.rio.clip([g], label_geotiffs_ds.rio.crs)

#     A = []
#     for x, y in zip(label_c.x, label_c.y):
#         tmp = label_c[1].sel(x=x, y=y).values
#         a = mc.MarkovChain().from_data(tmp)
#         A.append(a)

#     ind = np.where(np.array([len(a.observed_matrix) for a in A])==4)[0]
#     # np.array(A[ind].n_order_matrix)
#     if len(ind)>0:
#         PM.append(np.dstack([np.array(A[i].observed_p_matrix) for i in ind]).mean(axis=-1))
#         O2M.append(np.dstack([np.array(A[i].n_order_matrix(order=2)) for i in ind]).mean(axis=-1))
#         O3M.append(np.dstack([np.array(A[i].n_order_matrix(order=3)) for i in ind]).mean(axis=-1))
#     else:
#         PM.append(np.nan)
#         O2M.append(np.nan)
#         O3M.append(np.nan)


# np.savez('summaries/MR_transition_matrices.npz', MR_PM = PM, MR_O2M = O2M, MR_O3M = O3M)


# ## veg, water, sed, wood
# PM = []; O2M = []; O3M = []
# ## all - instantanues
# for counter,g in tqdm(enumerate(MRbudget_reaches_redo)):
#     print("Working on region {}".format(counter))
#     label_c = label_geotiffs_ds.rio.clip([g], label_geotiffs_ds.rio.crs)

#     A = []
#     for x, y in zip(label_c.x, label_c.y):
#         tmp = label_c[1].sel(x=x, y=y).values
#         a = mc.MarkovChain().from_data(tmp)
#         A.append(a)

#     ind = np.where(np.array([len(a.observed_matrix) for a in A])==4)[0]

#     try:
#         # np.array(A[ind].n_order_matrix)
#         PM.append(np.dstack([np.array(A[i].observed_p_matrix) for i in ind]).mean(axis=-1))
#         O2M.append(np.dstack([np.array(A[i].n_order_matrix(order=2)) for i in ind]).mean(axis=-1))
#         O3M.append(np.dstack([np.array(A[i].n_order_matrix(order=3)) for i in ind]).mean(axis=-1))
#     except:
#         PM.append(np.ones((4,4))*np.nan)
#         O2M.append(np.ones((4,4))*np.nan)
#         O3M.append(np.ones((4,4))*np.nan)

# np.savez('summaries/MR_transition_matrices_budgetreaches.npz', MR_PM = PM, MR_O2M = O2M, MR_O3M = O3M)



###############################
########### analysis of transition


# ## veg, water, sed, wood
# PM = []; O2M = []; O3M = []
# ## all - instantanues
# for counter,g in tqdm(enumerate(LRbudget_reaches_redo[42:])):
#     print("Working on region {}".format(counter))
#     label_c = label_geotiffs_ds.rio.clip([g], label_geotiffs_ds.rio.crs)

#     A = []
#     for x, y in zip(label_c.x, label_c.y):
#         tmp = label_c[1].sel(x=x, y=y).values
#         a = mc.MarkovChain().from_data(tmp)
#         A.append(a)

#     ind = np.where(np.array([len(a.observed_matrix) for a in A])==4)[0]
#     try:
#         # np.array(A[ind].n_order_matrix)
#         PM.append(np.dstack([np.array(A[i].observed_p_matrix) for i in ind]).mean(axis=-1))
#         O2M.append(np.dstack([np.array(A[i].n_order_matrix(order=2)) for i in ind]).mean(axis=-1))
#         O3M.append(np.dstack([np.array(A[i].n_order_matrix(order=3)) for i in ind]).mean(axis=-1))
#     except:
#         PM.append(np.ones((4,4))*np.nan)
#         O2M.append(np.ones((4,4))*np.nan)
#         O3M.append(np.ones((4,4))*np.nan)

# np.savez('summaries/LR_transition_matrices_budgetreaches_partial42_52.npz', LR_PM = PM, LR_O2M = O2M, LR_O3M = O3M)

# np.savez('summaries/LR_transition_matrices_budgetreaches_partial0_41.npz', LR_PM = PM, LR_O2M = O2M, LR_O3M = O3M)



##################################################################


dists = pd.read_csv('br_dists.csv')
LR = np.hstack((0,np.array(dists['LR'])))
MR = np.hstack((0,np.array(dists['MR'][:43])))


def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    Code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn


LR = rescale_array(LR,11,2)
MR = rescale_array(MR[::-1],12,20)

## budget reaches
brfile = '../results/LR/LR_wood/wood_detect/model1/LR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
LRbudget_reaches = gj['features']

LRbudget_reaches_redo = []
for b in LRbudget_reaches:
    LRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))

brfile = '../results/LR/LR_wood/wood_detect/model1/LR_budget_reaches_epsg4326.geojson'
with open(brfile) as f:
    gj = json.load(f)
LRbudget_reaches2 = gj['features']



## budget reaches
brfile = '../results/MR/MR_wood/wood_detect/model1/MR_budget_reaches.geojson'
with open(brfile) as f:
    gj = json.load(f)
MRbudget_reaches = gj['features']

MRbudget_reaches_redo = []
for b in MRbudget_reaches:
    MRbudget_reaches_redo.append(dict({'type': 'Polygon','coordinates': b['geometry']['coordinates'][0]}))

brfile = '../results/MR/MR_wood/wood_detect/model1/MR_budget_reaches_epsg4326.geojson'
with open(brfile) as f:
    gj = json.load(f)
MRbudget_reaches2 = gj['features']

############### MR

with np.load('summaries/MR_transition_matrices_budgetreaches.npz', allow_pickle=True) as dat:
    MR_tpm = dict()
    for k in dat.keys():
        MR_tpm[k] = dat[k]
    del dat

MR_TPM = []
for k in MR_tpm['MR_PM']:
    if np.isnan(k).any():
        tmp = np.ones((4,4))*np.nan
        MR_TPM.append(tmp)
    else:
        MR_TPM.append(k)


############### LR

with np.load('summaries/LR_transition_matrices_budgetreaches_partial0_41.npz', allow_pickle=True) as dat:
    LR_tpm = dict()
    for k in dat.keys():
        LR_tpm[k] = dat[k]
    del dat

LR_TPM = []
for k in LR_tpm['LR_PM']:
    if np.isnan(k).any():
        tmp = np.ones((4,4))*np.nan
        LR_TPM.append(tmp)
    else:
        LR_TPM.append(k)

with np.load('summaries/LR_transition_matrices_budgetreaches_partial42_52.npz', allow_pickle=True) as dat:
    LR_tpm = dict()
    for k in dat.keys():
        LR_tpm[k] = dat[k]
    del dat

for k in LR_tpm['LR_PM']:
    if np.isnan(k).any():
        tmp = np.ones((4,4))*np.nan
        LR_TPM.append(tmp)
    else:
        LR_TPM.append(k)

#############################

dMR_TPM = np.dstack(MR_TPM)
dLR_TPM = np.dstack(LR_TPM)


MR_ssv, U = np.linalg.eig(np.nanmedian(dMR_TPM,axis=-1).T)
##array([0.82745277, 0.0706406 , 0.31582526, 0.24454458])

LR_ssv, U = np.linalg.eig(np.nanmedian(dLR_TPM,axis=-1).T)
##array([0.79263494, 0.0898428 , 0.0898428 , 0.24794542])

MR_ssv = np.real(MR_ssv)/np.sum(np.real(MR_ssv))
LR_ssv = np.real(LR_ssv)/np.sum(np.real(LR_ssv))

#############################



plt.figure(figsize=(14,14))
plt.subplot(321)
g = sns.heatmap(np.nanmedian(dMR_TPM,axis=-1), annot = True, cmap ='plasma', vmax=1, vmin=0,
            linecolor ='black', linewidths = 1, cbar_kws={'label': 'Probability of transition'})

g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
plt.title('a) MR', loc='left'); 

plt.subplot(322)
g = sns.heatmap(np.nanmedian(dLR_TPM,axis=-1), annot = True, cmap ='plasma', vmax=1, vmin=0,
            linecolor ='black', linewidths = 1, cbar_kws={'label': 'Probability of transition'})

g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
plt.title('b) LR', loc='left'); 

# plt.savefig("summaries/median_reach_TPM_ds.png", dpi=300, bbox_inches="tight")
# plt.close()

fontsize = 9

plt.subplot(323)
g = sns.heatmap((np.nanmean(dMR_TPM,axis=-1)+np.nanmean(dLR_TPM,axis=-1))/2, annot = False, cmap ='plasma', vmax=1, vmin=0,
            linecolor ='black', linewidths = 1, cbar_kws={'label': 'Probability of transition'})

g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
plt.title('c)', loc='left'); 
plt.text(.05,.5,'Veg.\npersistence', color='w', fontsize=fontsize)
plt.text(.05,1.5,'Veg. erosion', color='w', fontsize=fontsize)
plt.text(.05,2.5,'Veg. burial/\n Sed. dep.', color='w', fontsize=fontsize)
plt.text(.05,3.5,'Wood dep./ \n Veg. removal', color='w', fontsize=fontsize)

plt.text(1.05,.5,'Veg. growth', color='w', fontsize=fontsize)
plt.text(1.05,1.5,'Water\npersistence', color='w', fontsize=fontsize)
plt.text(1.05,2.5,'Sed. dep.', color='w', fontsize=fontsize)
plt.text(1.05,3.5,'Wood dep.', color='w', fontsize=fontsize)

plt.text(2.05,.5,'Veg. growth', color='w', fontsize=fontsize)
plt.text(2.05,1.5,'Sed. erosion', color='w', fontsize=fontsize)
plt.text(2.05,2.5,'Sed.\npersistence', color='w', fontsize=fontsize)
plt.text(2.05,3.5,'Wood dep.', color='w', fontsize=fontsize)

plt.text(3.05,.5,'Wood\nocclusion', color='w', fontsize=fontsize)
plt.text(3.05,1.5,'Wood\nerosion', color='w', fontsize=fontsize)
plt.text(3.05,2.5,'Sed. dep/\n Wood burial', color='w', fontsize=fontsize)
plt.text(3.05,3.5,'Wood\npersistence', color='w', fontsize=fontsize)


# plt.figure(figsize=(12,8))
plt.subplot(324)
g = sns.heatmap(np.nanmedian(dMR_TPM,axis=-1)-np.nanmedian(dLR_TPM,axis=-1), annot = True, cmap ='bwr', vmax=.2, vmin=-.2,
            linecolor ='black', linewidths = 1, cbar_kws={'label': 'Probability divergence'})

g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
plt.title('d)', loc='left'); 

# plt.show()
# plt.savefig("summaries/median_reach_TPM_ds_diff.png", dpi=300, bbox_inches="tight")
# plt.close()

# plt.figure(figsize=(12,8))
plt.subplot(325)
g = sns.heatmap(np.nanstd(dMR_TPM,axis=-1)/np.nanmean(dMR_TPM,axis=-1), annot = True, cmap ='plasma', vmin=0, vmax=1.5,
            linecolor ='black', linewidths = 1, cbar_kws={'label': 'Spatial variability\n of transition (-)'})

g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
plt.title('e) MR', loc='left'); 

plt.subplot(326)
g = sns.heatmap(np.nanstd(dLR_TPM,axis=-1)/np.nanmean(dLR_TPM,axis=-1), annot = True, cmap ='plasma', vmin=0, vmax=1.5,
            linecolor ='black', linewidths = 1, cbar_kws={'label': 'Spatial variability\n of transition (-)'})

g.set_xticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
g.set_yticks([.5,1.5,2.5,3.5], ['Veg.','Water','Sediment', 'Wood'])
plt.title('f) LR', loc='left'); 

# plt.show()

plt.savefig("summaries/TPM_analysis.png", dpi=300, bbox_inches="tight")
plt.close()





############### MR

MR_wood_pers = [p[3,3] for p in MR_TPM]#
MR_sed_pers = [p[2,2] for p in  MR_TPM]#MR_tpm['MR_PM']]
MR_veg_pers = [p[0,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
MR_water_pers = [p[1,1] for p in  MR_TPM]#MR_tpm['MR_PM']]

MR_veg_erosion = [p[0,1] for p in  MR_TPM]#MR_tpm['MR_PM']]
MR_veg_burial = [p[0,2] for p in MR_TPM]# MR_tpm['MR_PM']]
MR_canopy_emerg = [p[0,3] for p in  MR_TPM]#MR_tpm['MR_PM']]

MR_veg_enc = [p[1,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
MR_sed_dep = [p[1,2] for p in  MR_TPM]#MR_tpm['MR_PM']]
MR_wood_dep_fromwater = [p[1,3] for p in  MR_TPM]#MR_tpm['MR_PM']]

MR_veg_growth = [p[2,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
MR_sed_erosion = [p[2,1] for p in  MR_TPM]#MR_tpm['MR_PM']]
MR_wood_dep_fromsed = [p[2,3] for p in  MR_TPM]#MR_tpm['MR_PM']]

MR_wood_occl = [p[3,0] for p in  MR_TPM]#MR_tpm['MR_PM']]
MR_wood_erosion = [p[3,1] for p in  MR_TPM]#MR_tpm['MR_PM']]
MR_wood_burial = [p[3,2] for p in  MR_TPM]#MR_tpm['MR_PM']]

MRxy = np.array([np.mean(np.mean(g['coordinates'],axis=0),axis=0).flatten() for g in MRbudget_reaches_redo])
MRx = MRxy[:,0]
MRy = MRxy[:,1]



############### LR

LR_wood_pers = [p[3,3] for p in LR_TPM]#
LR_sed_pers = [p[2,2] for p in  LR_TPM]#LR_tpm['LR_PM']]
LR_veg_pers = [p[0,0] for p in  LR_TPM]#LR_tpm['LR_PM']]
LR_water_pers = [p[1,1] for p in  LR_TPM]#LR_tpm['LR_PM']]

LR_veg_erosion = [p[0,1] for p in  LR_TPM]#LR_tpm['LR_PM']]
LR_veg_burial = [p[0,2] for p in LR_TPM]# LR_tpm['LR_PM']]
LR_canopy_emerg = [p[0,3] for p in  LR_TPM]#LR_tpm['LR_PM']]

LR_veg_enc = [p[1,0] for p in  LR_TPM]#LR_tpm['LR_PM']]
LR_sed_dep = [p[1,2] for p in  LR_TPM]#LR_tpm['LR_PM']]
LR_wood_dep_fromwater = [p[1,3] for p in  LR_TPM]#LR_tpm['LR_PM']]

LR_veg_growth = [p[2,0] for p in  LR_TPM]#LR_tpm['LR_PM']]
LR_sed_erosion = [p[2,1] for p in  LR_TPM]#LR_tpm['LR_PM']]
LR_wood_dep_fromsed = [p[2,3] for p in  LR_TPM]#LR_tpm['LR_PM']]

LR_wood_occl = [p[3,0] for p in  LR_TPM]#LR_tpm['LR_PM']]
LR_wood_erosion = [p[3,1] for p in  LR_TPM]#LR_tpm['LR_PM']]
LR_wood_burial = [p[3,2] for p in  LR_TPM]#LR_tpm['LR_PM']]

LRxy = np.array([np.mean(np.mean(g['coordinates'],axis=0),axis=0).flatten() for g in LRbudget_reaches_redo])
LRx = LRxy[:,0]
LRy = LRxy[:,1]




##################################################################

dat = {
    ## persistence
    "wood pers.": np.array(MR_wood_pers),
    "sed. pers.": np.array(MR_sed_pers),
    "veg. pers.": np.array(MR_veg_pers),
    "water pers.": np.array(MR_water_pers),

    ## remov veg
    "veg. erosion (water)": np.array(MR_veg_erosion),    
    "veg. erosion (sed.)": np.array(MR_veg_burial),   

    ## added veg
    "veg. growth (water)": np.array(MR_veg_enc),
    "veg. growth (sed.)": np.array(MR_veg_growth),

    ## added wood
    "wood dep. (water)": np.array(MR_wood_dep_fromwater),      
    "wood dep. (sed.)": np.array(MR_wood_dep_fromsed),  
    "wood uncovering (veg.)": np.array(MR_canopy_emerg),

    ## sed
    "sed. erosion": np.array(MR_sed_erosion),    
    "sed. dep.": np.array(MR_sed_dep),      

    ## remove wood
    "wood covering (veg.)": np.array(MR_wood_occl),
    "wood erosion": np.array(MR_wood_erosion),      
    "wood burial": np.array(MR_wood_burial),    
}

dat = pd.DataFrame.from_dict(dat, orient='index')

dat2 = dat.T.dropna()

add_wood = dat2[ "wood dep. (water)"].values+ dat2["wood dep. (sed.)"].values+ dat2["wood uncovering (veg.)"].values
remove_wood = dat2["wood covering (veg.)"].values+ dat2["wood erosion"].values+ dat2["wood burial"].values

add_veg = dat2["veg. growth (water)"].values+ dat2[ "veg. growth (sed.)"].values
remove_veg = dat2["veg. erosion (water)"].values+ dat2["veg. erosion (sed.)"].values


MRi  = np.interp(np.linspace(0,len(MR),len(dat2)), np.arange(len(MR)), MR)
#########################################



LRdat = {
    ## persistence
    "wood pers.": np.array(LR_wood_pers),
    "sed. pers.": np.array(LR_sed_pers),
    "veg. pers.": np.array(LR_veg_pers),
    "water pers.": np.array(LR_water_pers),

    ## remov veg
    "veg. erosion (water)": np.array(LR_veg_erosion),    
    "veg. erosion (sed.)": np.array(LR_veg_burial),   

    ## added veg
    "veg. growth (water)": np.array(LR_veg_enc),
    "veg. growth (sed.)": np.array(LR_veg_growth),

    ## added wood
    "wood dep. (water)": np.array(LR_wood_dep_fromwater),      
    "wood dep. (sed.)": np.array(LR_wood_dep_fromsed),  
    "wood uncovering (veg.)": np.array(LR_canopy_emerg),

    ## sed
    "sed. erosion": np.array(LR_sed_erosion),    
    "sed. dep.": np.array(LR_sed_dep),      

    ## remove wood
    "wood covering (veg.)": np.array(LR_wood_occl),
    "wood erosion": np.array(LR_wood_erosion),      
    "wood burial": np.array(LR_wood_burial),    
}

LRdat = pd.DataFrame.from_dict(LRdat, orient='index')

LRdat2 = LRdat.T.dropna()

LRadd_wood = LRdat2[ "wood dep. (water)"].values+ LRdat2["wood dep. (sed.)"].values+ LRdat2["wood uncovering (veg.)"].values
LRremove_wood = LRdat2["wood covering (veg.)"].values+ LRdat2["wood erosion"].values+ LRdat2["wood burial"].values

LRadd_veg = LRdat2["veg. growth (water)"].values+ LRdat2[ "veg. growth (sed.)"].values
LRremove_veg = LRdat2["veg. erosion (water)"].values+ LRdat2["veg. erosion (sed.)"].values


LRi  = np.interp(np.linspace(0,len(LR),len(LRdat2)), np.arange(len(LR)), LR)



###############################################




plt.figure(figsize=(12,18))

plt.subplot(621)
plt.plot(MRi, add_wood/3,'-', color='brown', label="Processes that add wood)")
plt.plot(MRi, remove_wood/3,'--', color='brown', label="Wood removal processes")
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(0,1)
plt.title('a)', loc='left')

plt.subplot(623)
plt.plot(MRi, (add_wood/3) - (remove_wood/3), 'k', label="Net probability")
plt.plot(MRi, dat2["wood pers."].values,':', lw=2, color='brown', label="Persistent wood")
plt.axhline(0)
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(-1,1)
plt.title('b)', loc='left')

plt.subplot(625)
plt.plot(MRi,dat2["sed. dep."].values,'-', color='gold', label="Processes that add sediment)")
plt.plot(MRi,dat2["sed. erosion"].values,'--', color='gold', label="Sediment removal processes")
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(0,1)
plt.title('c)', loc='left')

plt.subplot(627)
plt.plot(MRi, (dat2["sed. dep."].values) - (dat2["sed. erosion"].values), 'k', label="Net probability")
plt.plot(MRi, dat2["sed. pers."].values,':', lw=2, color='gold', label="Persistent sediment")
plt.axhline(0)
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(-1,1)
plt.title('d)', loc='left')

plt.subplot(629)
plt.plot(MRi, add_veg/2,'-', color='lawngreen', label="Processes that add veg.)")
plt.plot(MRi, remove_veg/2,'--', color='lawngreen', label="Veg. removal processes")
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(0,1)
plt.title('e)', loc='left')

plt.subplot(6,2,11)
plt.plot(MRi, (add_veg/2) - (remove_veg/2), 'k', label="Net probability")
plt.plot(MRi, dat2["veg. pers."].values,':', lw=2, color='lawngreen', label="Persistent veg.")
plt.axhline(0)
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(-1,1)
plt.xlabel('River kilometer')
plt.ylabel('Probability')
plt.title('f)', loc='left')


#### LR

plt.subplot(622)
plt.plot(LRi, LRadd_wood/3,'-', color='brown', label="Processes that add wood)")
plt.plot(LRi, LRremove_wood/3,'--', color='brown', label="Wood removal processes")
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(0,1)

plt.subplot(624)
plt.plot(LRi, (LRadd_wood/3) - (LRremove_wood/3), 'k', label="Net probability")
plt.plot(LRi, LRdat2["wood pers."].values,':', lw=2, color='brown', label="Persistent wood")
plt.axhline(0)
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(-1,1)

plt.subplot(626)
plt.plot(LRi,LRdat2["sed. dep."].values,'-', color='gold', label="Processes that add sediment)")
plt.plot(LRi,LRdat2["sed. erosion"].values,'--', color='gold', label="Sediment removal processes")
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(0,1)

plt.subplot(628)
plt.plot(LRi, (LRdat2["sed. dep."].values) - (LRdat2["sed. erosion"].values), 'k', label="Net probability")
plt.plot(LRi, LRdat2["sed. pers."].values,':', lw=2, color='gold', label="Persistent sediment")
plt.axhline(0)
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(-1,1)

plt.subplot(6,2,10)
plt.plot(LRi, LRadd_veg/2,'-', color='lawngreen', label="Processes that add veg.)")
plt.plot(LRi, LRremove_veg/2,'--', color='lawngreen', label="Veg. removal processes")
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(0,1)

plt.subplot(6,2,12)
plt.plot(LRi, (LRadd_veg/2) - (LRremove_veg/2), 'k', label="Net probability")
plt.plot(LRi, LRdat2["veg. pers."].values,':', lw=2, color='lawngreen', label="Persistent veg.")
plt.axhline(0)
plt.legend(fontsize=7)
plt.gca().invert_xaxis()
plt.ylim(-1,1)
plt.xlabel('River kilometer')
plt.ylabel('Probability')


# plt.show()
plt.savefig("summaries/MR_LR_transition_dynamics.png", dpi=300, bbox_inches="tight")
plt.close()





# cmap = ListedColormap(["black","lawngreen", "blue", "gold", "brown"])
# colors = ["lawngreen", "blue", "gold", "brown"]

cmap = ListedColormap(["black", "brown"])
colors = ["brown"]

ds =  [str(i)[:4] for i in MRi] 
width = 0.6

fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize=(18,12))

tmp = dat2[[ "wood pers."]]
# tmp = dat2[["veg. pers.", "water pers.", "sed. pers.", "wood pers."]]
keys = list(tmp.keys())
wc = {}
for k in range(tmp.to_numpy().shape[1]):
    wc[keys[k]] = tmp.to_numpy()[:,k]

bottom = np.zeros(tmp.to_numpy().shape[0])
for counter, (label, w) in enumerate(wc.items()):
    # print(counter)
    p = ax[0].bar(ds, w, width, label=label, bottom=bottom, color=colors[counter])
    bottom += w
    counter += 1
ax[0].set_title("a) Wood persistence",loc='left')
ax[0].legend(loc="upper left")
# ax[0].invert_xaxis()
ax[0].set_ylim(0,1)

colors2 = ['cornflowerblue','goldenrod','yellowgreen']
tmp = dat2[["wood dep. (water)", "wood dep. (sed.)", "wood uncovering (veg.)"]]
keys = list(tmp.keys())
wc = {}
for k in range(tmp.to_numpy().shape[1]):
    wc[keys[k]] = tmp.to_numpy()[:,k]

bottom = np.zeros(tmp.to_numpy().shape[0])
for counter, (label, w) in enumerate(wc.items()):
    p = ax[1].bar(ds, w, width, label=label, bottom=bottom, color=colors2[counter])
    bottom += w
    counter += 1
ax[1].set_title("b) Wood addition",loc='left')
ax[1].legend(loc="upper left")
# ax[1].invert_xaxis()
ax[1].set_ylim(0,1)

colors3 = ['yellowgreen','lightsteelblue','khaki']
tmp = dat2[["wood covering (veg.)", "wood erosion", "wood burial"]]
keys = list(tmp.keys())
wc = {}
for k in range(tmp.to_numpy().shape[1]):
    wc[keys[k]] = tmp.to_numpy()[:,k]

bottom = np.zeros(tmp.to_numpy().shape[0])
for counter, (label, w) in enumerate(wc.items()):
    # print(counter)
    p = ax[2].bar(ds, w, width, label=label, bottom=bottom, color=colors3[counter])
    bottom += w
    counter += 1
ax[2].set_title("c) Wood removal",loc='left')
ax[2].legend(loc="upper left")
# ax[2].invert_xaxis()
ax[2].set_ylim(0,1)

# colors4 = ['deepskyblue','gold']
# tmp = dat2[["sed. erosion", "sed. dep."]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[3].bar(ds, w, width, label=label, bottom=bottom, color=colors4[counter])
#     bottom += w
#     counter += 1
# ax[3].set_title("d) Sediment",loc='left')
# ax[3].legend(loc="upper left")
# # ax[3].invert_xaxis()
# ax[3].set_ylim(0,2)

# colors5 = ['deepskyblue','gold', 'palegreen', 'limegreen']
# tmp = dat2[["veg. erosion (water)", "veg. erosion (sed.)", "veg. growth (water)",  "veg. growth (sed.)"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[4].bar(ds, w, width, label=label, bottom=bottom, color=colors5[counter])
#     bottom += w
#     counter += 1
# ax[4].set_title("e) Vegetation",loc='left')
# ax[4].legend(loc="upper left")
# # ax[4].invert_xaxis()
# ax[4].set_ylim(0,2)
# plt.xlabel('River kilometer')
# plt.ylabel('Transition probability')

# plt.show()
plt.savefig("summaries/MR_transitions_ds_v2.png", dpi=300, bbox_inches="tight")
plt.close()



# cmap = ListedColormap(["black","lawngreen", "blue", "gold", "brown"])
# colors = ["lawngreen", "blue", "gold", "brown"]

cmap = ListedColormap(["black", "brown"])
colors = ["brown"]

ds =  [str(i)[:4] for i in LRi] 
width = 0.6

fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize=(18,12))

tmp = LRdat2[[ "wood pers."]]
# tmp = dat2[["veg. pers.", "water pers.", "sed. pers.", "wood pers."]]
keys = list(tmp.keys())
wc = {}
for k in range(tmp.to_numpy().shape[1]):
    wc[keys[k]] = tmp.to_numpy()[:,k]

bottom = np.zeros(tmp.to_numpy().shape[0])
for counter, (label, w) in enumerate(wc.items()):
    # print(counter)
    p = ax[0].bar(ds, w, width, label=label, bottom=bottom, color=colors[counter])
    bottom += w
    counter += 1
# ax[0].set_title("a) Wood persistence",loc='left')
ax[0].legend(loc="upper left")
# ax[0].invert_xaxis()
ax[0].set_ylim(0,1)

colors2 = ['cornflowerblue','goldenrod','yellowgreen']
tmp = LRdat2[["wood dep. (water)", "wood dep. (sed.)", "wood uncovering (veg.)"]]
keys = list(tmp.keys())
wc = {}
for k in range(tmp.to_numpy().shape[1]):
    wc[keys[k]] = tmp.to_numpy()[:,k]

bottom = np.zeros(tmp.to_numpy().shape[0])
for counter, (label, w) in enumerate(wc.items()):
    p = ax[1].bar(ds, w, width, label=label, bottom=bottom, color=colors2[counter])
    bottom += w
    counter += 1
ax[1].set_title("b) Wood addition",loc='left')
ax[1].legend(loc="upper left")
# ax[1].invert_xaxis()
ax[1].set_ylim(0,1)

colors3 = ['yellowgreen','lightsteelblue','khaki']
tmp = LRdat2[["wood covering (veg.)", "wood erosion", "wood burial"]]
keys = list(tmp.keys())
wc = {}
for k in range(tmp.to_numpy().shape[1]):
    wc[keys[k]] = tmp.to_numpy()[:,k]

bottom = np.zeros(tmp.to_numpy().shape[0])
for counter, (label, w) in enumerate(wc.items()):
    # print(counter)
    p = ax[2].bar(ds, w, width, label=label, bottom=bottom, color=colors3[counter])
    bottom += w
    counter += 1
ax[2].set_title("c) Wood removal",loc='left')
ax[2].legend(loc="upper left")
# ax[2].invert_xaxis()
ax[2].set_ylim(0,1)

# colors4 = ['deepskyblue','gold']
# tmp = dat2[["sed. erosion", "sed. dep."]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[3].bar(ds, w, width, label=label, bottom=bottom, color=colors4[counter])
#     bottom += w
#     counter += 1
# ax[3].set_title("d) Sediment",loc='left')
# ax[3].legend(loc="upper left")
# # ax[3].invert_xaxis()
# ax[3].set_ylim(0,2)

# colors5 = ['deepskyblue','gold', 'palegreen', 'limegreen']
# tmp = dat2[["veg. erosion (water)", "veg. erosion (sed.)", "veg. growth (water)",  "veg. growth (sed.)"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[4].bar(ds, w, width, label=label, bottom=bottom, color=colors5[counter])
#     bottom += w
#     counter += 1
# ax[4].set_title("e) Vegetation",loc='left')
# ax[4].legend(loc="upper left")
# # ax[4].invert_xaxis()
# ax[4].set_ylim(0,2)
# plt.xlabel('River kilometer')
# plt.ylabel('Transition probability')

# plt.show()
plt.savefig("summaries/LR_transitions_ds_v2.png", dpi=300, bbox_inches="tight")
plt.close()




# ds =  [str(i)[:4] for i in LRi] 
# width = 0.6

# fig, ax = plt.subplots(nrows = 5, ncols = 1, figsize=(15,20))

# tmp = LRdat2[["veg. pers.", "water pers.", "sed. pers.", "wood pers."]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[0].bar(ds, w, width, label=label, bottom=bottom, color=colors[counter])
#     bottom += w
#     counter += 1
# ax[0].set_title("b) Persistence",loc='left')
# ax[0].legend(loc="upper left")
# # ax[0].invert_xaxis()
# ax[0].set_ylim(0,2)

# # ax[0].tick_params(left = False, right = False , labelleft = False , 
# #                 labelbottom = True, bottom = True) 

# colors2 = ['cornflowerblue','goldenrod','yellowgreen']
# tmp = LRdat2[["wood dep. (water)", "wood dep. (sed.)", "wood uncovering (veg.)"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     p = ax[1].bar(ds, w, width, label=label, bottom=bottom, color=colors2[counter])
#     bottom += w
#     counter += 1
# ax[1].set_title("b) Added wood",loc='left')
# ax[1].legend(loc="upper left")
# # ax[1].invert_xaxis()
# ax[1].set_ylim(0,2)
# # ax[1].tick_params(left = False, right = False , labelleft = False , 
# #                 labelbottom = True, bottom = True) 

# colors3 = ['yellowgreen','lightsteelblue','khaki']
# tmp = LRdat2[["wood covering (veg.)", "wood erosion", "wood burial"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[2].bar(ds, w, width, label=label, bottom=bottom, color=colors3[counter])
#     bottom += w
#     counter += 1
# ax[2].set_title("c) Removed wood",loc='left')
# ax[2].legend(loc="upper left")
# # ax[2].invert_xaxis()
# ax[2].set_ylim(0,2)
# # ax[2].tick_params(left = False, right = False , labelleft = False , 
# #                 labelbottom = True, bottom = True) 

# colors4 = ['deepskyblue','gold']
# tmp = LRdat2[["sed. erosion", "sed. dep."]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[3].bar(ds, w, width, label=label, bottom=bottom, color=colors4[counter])
#     bottom += w
#     counter += 1
# ax[3].set_title("d) Sediment",loc='left')
# ax[3].legend(loc="upper left")
# # ax[3].invert_xaxis()
# ax[3].set_ylim(0,2)
# # ax[3].tick_params(left = False, right = False , labelleft = False , 
# #                 labelbottom = True, bottom = True) 

# colors5 = ['deepskyblue','gold', 'palegreen', 'limegreen']
# tmp = LRdat2[["veg. erosion (water)", "veg. erosion (sed.)", "veg. growth (water)",  "veg. growth (sed.)"]]
# keys = list(tmp.keys())
# wc = {}
# for k in range(tmp.to_numpy().shape[1]):
#     wc[keys[k]] = tmp.to_numpy()[:,k]

# bottom = np.zeros(tmp.to_numpy().shape[0])
# for counter, (label, w) in enumerate(wc.items()):
#     # print(counter)
#     p = ax[4].bar(ds, w, width, label=label, bottom=bottom, color=colors5[counter])
#     bottom += w
#     counter += 1
# ax[4].set_title("e) Vegetation",loc='left')
# ax[4].legend(loc="upper left")
# # ax[4].invert_xaxis()
# ax[4].set_ylim(0,2)
# # ax[4].tick_params(left = False, right = False , labelleft = False , 
# #                 labelbottom = True, bottom = True) 
# plt.xlabel('River kilometer')
# # plt.ylabel('Transition probability')

# # plt.show()
# plt.savefig("summaries/LR_transitions_ds.png", dpi=300, bbox_inches="tight")
# plt.close()
