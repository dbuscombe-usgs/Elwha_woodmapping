
import numpy as np 
from matplotlib import pyplot as plt 
from glob import glob 
import pandas as pd


npz_wood = sorted(glob("../gym/v7_wood/zenodo_release_segformer/*.npz"))
npz_water = sorted(glob("../gym/v5/weights/water/*.npz"))
npz_veg = sorted(glob("../gym/v5/weights/veg/*.npz"))
npz_all = sorted(glob("../gym/v9_all/*.npz"))
npz_sed = sorted(glob("../gym/v8_sed/*.npz"))

# csv_wood = (glob("../gym/v5/modelOut/wood/v1/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/wood/v2/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/wood/v3/*per_sample_val.csv"))
# csv_water = (glob("../gym/v5/modelOut/water/v1/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/water/v2/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/water/v3/*per_sample_val.csv"))
# csv_veg = (glob("../gym/v5/modelOut/veg/v1/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/veg/v2/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/veg/v3/*per_sample_val.csv"))
# csv_all = (glob("../gym/v5/modelOut/all/v1/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/all/v2/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/all/v3/*per_sample_val.csv"))


########################################
plt.figure(figsize=(16,16))
plt.subplots_adjust(wspace=0.3, hspace=0.3)


L=[]; VL=[]
for file in npz_wood:
    with np.load(file) as f:
        loss = f['loss']
        val_loss = f['val_loss']
    L.append(loss)
    VL.append(val_loss)

plt.subplot(221)
plt.semilogx(L[0], 'm-', label='Model 1, train')
plt.plot(L[1], 'g-', label='Model 2, train')
plt.plot(L[2], 'b-', label='Model 3, train')
plt.plot(VL[0], 'm--', label='Model 1, val.')
plt.plot(VL[1], 'g--', label='Model 2, val.')
plt.plot(VL[2], 'b--', label='Model 3, val.')
plt.ylabel("Loss (non-dim.)")
plt.title('a) Wood',loc='left')
# plt.text(10,.6,'Wood')
plt.ylim(0,0.8)
plt.xlim(0,100)
plt.xlabel("Training epoch")
plt.legend()

L=[]; VL=[]
for file in npz_sed:
    with np.load(file) as f:
        loss = f['loss']
        val_loss = f['val_loss']
    L.append(loss)
    VL.append(val_loss)

plt.subplot(222)
plt.semilogx(L[0], 'm-', label='Model 1, train')
plt.plot(L[1], 'g-', label='Model 2, train')
plt.plot(L[2], 'b-', label='Model 3, train')
plt.plot(VL[0], 'm--', label='Model 1, val.')
plt.plot(VL[1], 'g--', label='Model 2, val.')
plt.plot(VL[2], 'b--', label='Model 3, val.')
plt.ylabel("Loss (non-dim.)")
plt.title('b) Sediment',loc='left')
# plt.text(10,.6,'Wood')
plt.ylim(0,0.8)
plt.xlim(0,100)
plt.xlabel("Training epoch")



L=[]; VL=[]
for file in npz_water:
    with np.load(file) as f:
        loss = f['loss']
        val_loss = f['val_loss']
    L.append(loss)
    VL.append(val_loss)

plt.subplot(223)
plt.semilogx(L[0], 'm-', label='Model 1, train')
plt.plot(L[1], 'g-', label='Model 2, train')
plt.plot(L[2], 'b-', label='Model 3, train')
plt.plot(VL[0], 'm--', label='Model 1, val.')
plt.plot(VL[1], 'g--', label='Model 2, val.')
plt.plot(VL[2], 'b--', label='Model 3, val.')
plt.xlabel("Training epoch")
plt.ylabel("Loss (non-dim.)")
plt.title('c) Water',loc='left')
# plt.text(10,.6,'Water')
plt.ylim(0,0.8)
plt.xlim(0,100)

L=[]; VL=[]
for file in npz_veg:
    with np.load(file) as f:
        loss = f['loss']
        val_loss = f['val_loss']
    L.append(loss)
    VL.append(val_loss)

plt.subplot(224)
plt.semilogx(L[0], 'm-', label='Model 1, train')
plt.plot(L[1], 'g-', label='Model 2, train')
plt.plot(L[2], 'b-', label='Model 3, train')
plt.plot(VL[0], 'm--', label='Model 1, val.')
plt.plot(VL[1], 'g--', label='Model 2, val.')
plt.plot(VL[2], 'b--', label='Model 3, val.')
plt.xlabel("Training epoch")
plt.ylabel("Loss (non-dim.)")
plt.title('d) Other',loc='left')
# plt.text(10,.6,'Veg')
# plt.legend()
plt.ylim(0,0.8)
plt.xlim(0,100)

# plt.show()
plt.savefig("model_training_comparison.png", dpi=300, bbox_inches="tight")
plt.close()



npz_all_N = sorted(glob("../gym/v9_all/model_comparison_N/*.npz"))

N = [1999,2999,4382] #999,

N = np.array(N)

L=[]; VL=[]
for file in npz_all:
    with np.load(file) as f:
        loss = f['loss']
        val_loss = f['val_loss']
    L.append(loss)
    VL.append(val_loss)


fL=[]; fVL=[]
for file in npz_all_N:
    with np.load(file) as f:
        loss = f['loss']
        val_loss = f['val_loss']
    fL.append(loss)
    fVL.append(val_loss)


min_full_L = np.min([np.min(v) for v in fL])
min_full_VL = np.min([np.min(v) for v in fVL])


csv_all = sorted(glob("../gym/v9_all/model_comparison_N/*per_sample_val.csv")) + glob("../gym/v9_all/*v3*per_sample_val.csv")

fwiou=[]; miou = []; mcc = []
for k in csv_all:
    miou.append(pd.read_csv(k).mean()['MeanIntersectionOverUnion'])
    fwiou.append(pd.read_csv(k).mean()['Frequency_Weighted_Intersection_over_Union'])
    mcc.append(pd.read_csv(k).mean()['MatthewsCorrelationCoefficient'])

csv_all = sorted(glob("../gym/v9_all/model_comparison_N/*per_sample_train.csv")) + glob("../gym/v9_all/*v3*per_sample_train.csv")

fwiou2=[]; miou2 = []; mcc2 = []
for k in csv_all:
    miou2.append(pd.read_csv(k).mean()['MeanIntersectionOverUnion'])
    fwiou2.append(pd.read_csv(k).mean()['Frequency_Weighted_Intersection_over_Union'])
    mcc2.append(pd.read_csv(k).mean()['MatthewsCorrelationCoefficient'])


from brokenaxes import brokenaxes



########################################
# plt.figure(figsize=(8,16))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(111)
plt.semilogx(L[0], 'm-', label='Model 1, train')
plt.plot(L[1], 'g-', label='Model 2, train')
plt.plot(L[2], 'b-', label='Model 3, train')
plt.plot(L[3], 'c-', label='Model 4, train')
plt.plot(L[4], '-',color=[.75,.75,.75],  label='Model 5, train')

plt.plot(VL[0], 'm--', label='Model 1, val.')
plt.plot(VL[1], 'g--', label='Model 2, val.')
plt.plot(VL[2], 'b--', label='Model 3, val.')
plt.plot(VL[3], 'c--', label='Model 4, val.')
plt.plot(VL[4], '--',color=[.75,.75,.75],  label='Model 5, val.')
# plt.xlabel("Training epoch")
plt.ylabel("Loss (non-dim.)")
plt.title('a)',loc='left')
plt.legend()
plt.ylim(0,0.6)
plt.xlim(0,30)
# plt.text(10,.6,'Four classes')
plt.xlabel("Training epoch")
# plt.show()
plt.savefig("model_training.png", dpi=300, bbox_inches="tight")
plt.close()


plt.subplot(111)
bax = brokenaxes( ylims=((0.0875, .1), (.2, .206)), hspace=.1)
bax.plot(N, np.hstack(([np.min(v) for v in fL[1:]], min_full_L)), '-o', color=[.75,.75,.75], label='Train')
bax.plot(N, np.hstack(([np.min(v) for v in fVL[1:]], min_full_VL)), 'm--s',  label='Validation')  
bax.legend()       
# plt.ylabel("Mimimum loss (non-dim.)")
# plt.xlabel("Number of labeled training images")
plt.title('b)',loc='left')
# plt.show()
plt.savefig("model_training_A.png", dpi=300, bbox_inches="tight")
plt.close()

plt.subplot(111)
# bax = brokenaxes( ylims=((0.0875, .1), (.2, .206)), hspace=.1)
plt.plot(N, np.hstack(([np.min(v) for v in fL[1:]], min_full_L)), '-o', color=[.75,.75,.75], label='Train')
plt.plot(N, np.hstack(([np.min(v) for v in fVL[1:]], min_full_VL)), 'm--s',  label='Validation')  
plt.legend()       
# plt.ylabel("Mimimum loss (non-dim.)")
# plt.xlabel("Number of labeled training images")
plt.title('b)',loc='left')
# plt.show()
plt.savefig("model_training_B.png", dpi=300, bbox_inches="tight")
plt.close()


plt.subplot(111)
plt.plot(N*.7, sorted(miou2)[1:], '-o', color=[.75,.75,.75], label='mIOU, train')
plt.plot(N*.3, sorted(miou)[1:], '--o', color=[.75,.75,.75], label='mIOU, val')
plt.plot(N*.7, sorted(fwiou2)[1:], 'm-s', label='fwIOU, train')
plt.plot(N*.3, sorted(fwiou)[1:], 'm--s', label='fwIOU, val')
plt.plot(N*.7, sorted(mcc2)[1:], 'b-p',label='MCC, train')
plt.plot(N*.3, sorted(mcc)[1:], 'b--p', label='MCC, val')

plt.legend()
plt.ylabel("Metric (non-dim.)")
plt.xlabel("Number of labeled training images")
plt.title('c)',loc='left')

# plt.show()
plt.savefig("model_training_N.png", dpi=300, bbox_inches="tight")
plt.close()


# # plt.show()

# val1_wood = [pd.read_csv(csv_wood[0]).mean()['MeanIntersectionOverUnion'],
#                 pd.read_csv(csv_wood[1]).mean()['MeanIntersectionOverUnion'],
#                 pd.read_csv(csv_wood[2]).mean()['MeanIntersectionOverUnion']]


# val2_wood = [pd.read_csv(csv_wood[0]).mean()['Frequency_Weighted_Intersection_over_Union'],
#                 pd.read_csv(csv_wood[1]).mean()['Frequency_Weighted_Intersection_over_Union'],
#                 pd.read_csv(csv_wood[2]).mean()['Frequency_Weighted_Intersection_over_Union']]


# val3_wood = [pd.read_csv(csv_wood[0]).mean()['MatthewsCorrelationCoefficient'],
#                 pd.read_csv(csv_wood[1]).mean()['MatthewsCorrelationCoefficient'],
#                 pd.read_csv(csv_wood[2]).mean()['MatthewsCorrelationCoefficient']]

# plt.subplot(422)
# plt.bar([1,2,3], [np.mean(val1_wood), np.mean(val2_wood), np.mean(val3_wood)] )
# ax=plt.gca()
# ax.set_xticks([1,2,3])
# ax.set_xticklabels(["mIOU","fwIOU","MCC"])
# plt.ylim(0,1)
# plt.ylabel("Score (non-dim)")
# plt.title('b)',loc='left')
# plt.axhline(y=0.5, color=[.5,.5,.5], linestyle=':')

# val1_water = [pd.read_csv(csv_water[0]).mean()['MeanIntersectionOverUnion'],
#                 pd.read_csv(csv_water[1]).mean()['MeanIntersectionOverUnion'],
#                 pd.read_csv(csv_water[2]).mean()['MeanIntersectionOverUnion']]


# val2_water = [pd.read_csv(csv_water[0]).mean()['Frequency_Weighted_Intersection_over_Union'],
#                 pd.read_csv(csv_water[1]).mean()['Frequency_Weighted_Intersection_over_Union'],
#                 pd.read_csv(csv_water[2]).mean()['Frequency_Weighted_Intersection_over_Union']]


# val3_water = [pd.read_csv(csv_water[0]).mean()['MatthewsCorrelationCoefficient'],
#                 pd.read_csv(csv_water[1]).mean()['MatthewsCorrelationCoefficient'],
#                 pd.read_csv(csv_water[2]).mean()['MatthewsCorrelationCoefficient']]

# plt.subplot(424)
# plt.bar([1,2,3], [np.mean(val1_water), np.mean(val2_water), np.mean(val3_water)] )
# ax=plt.gca()
# ax.set_xticks([1,2,3])
# ax.set_xticklabels(["mIOU","fwIOU","MCC"])
# plt.ylim(0,1)
# plt.ylabel("Score (non-dim)")
# plt.title('d)',loc='left')
# plt.axhline(y=0.5, color=[.5,.5,.5], linestyle=':')

# val1_veg = [pd.read_csv(csv_veg[0]).mean()['MeanIntersectionOverUnion'],
#                 pd.read_csv(csv_veg[1]).mean()['MeanIntersectionOverUnion'],
#                 pd.read_csv(csv_veg[2]).mean()['MeanIntersectionOverUnion']]


# val2_veg = [pd.read_csv(csv_veg[0]).mean()['Frequency_Weighted_Intersection_over_Union'],
#                 pd.read_csv(csv_veg[1]).mean()['Frequency_Weighted_Intersection_over_Union'],
#                 pd.read_csv(csv_veg[2]).mean()['Frequency_Weighted_Intersection_over_Union']]


# val3_veg = [pd.read_csv(csv_veg[0]).mean()['MatthewsCorrelationCoefficient'],
#                 pd.read_csv(csv_veg[1]).mean()['MatthewsCorrelationCoefficient'],
#                 pd.read_csv(csv_veg[2]).mean()['MatthewsCorrelationCoefficient']]

# plt.subplot(426)
# plt.bar([1,2,3], [np.mean(val1_veg), np.mean(val2_veg), np.mean(val3_veg)] )
# ax=plt.gca()
# ax.set_xticks([1,2,3])
# ax.set_xticklabels(["mIOU","fwIOU","MCC"])
# plt.ylim(0,1)
# plt.ylabel("Score (non-dim)")
# plt.title('f)',loc='left')
# plt.axhline(y=0.5, color=[.5,.5,.5], linestyle=':')

# val1_all = [pd.read_csv(csv_all[0]).mean()['MeanIntersectionOverUnion'],
#                 pd.read_csv(csv_all[1]).mean()['MeanIntersectionOverUnion'],
#                 pd.read_csv(csv_all[2]).mean()['MeanIntersectionOverUnion']]


# val2_all = [pd.read_csv(csv_all[0]).mean()['Frequency_Weighted_Intersection_over_Union'],
#                 pd.read_csv(csv_all[1]).mean()['Frequency_Weighted_Intersection_over_Union'],
#                 pd.read_csv(csv_all[2]).mean()['Frequency_Weighted_Intersection_over_Union']]


# val3_all = [pd.read_csv(csv_all[0]).mean()['MatthewsCorrelationCoefficient'],
#                 pd.read_csv(csv_all[1]).mean()['MatthewsCorrelationCoefficient'],
#                 pd.read_csv(csv_all[2]).mean()['MatthewsCorrelationCoefficient']]

# plt.subplot(428)
# plt.bar([1,2,3], [np.mean(val1_all), np.mean(val2_all), np.mean(val3_all)] )
# ax=plt.gca()
# ax.set_xticks([1,2,3])
# ax.set_xticklabels(["mIOU","fwIOU","MCC"])
# plt.ylim(0,1)
# plt.ylabel("Score (non-dim)")
# plt.title('h)',loc='left')
# plt.axhline(y=0.5, color=[.5,.5,.5], linestyle=':')

# # plt.show()

# plt.savefig("model_skill_valset.png", dpi=300, bbox_inches="tight")
# plt.close()

