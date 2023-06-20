
import numpy as np 
from matplotlib import pyplot as plt 
from glob import glob 
import pandas as pd


npz_wood = sorted(glob("../gym/v5/weights/wood/*.npz"))
npz_water = sorted(glob("../gym/v5/weights/water/*.npz"))
npz_veg = sorted(glob("../gym/v5/weights/veg/*.npz"))
npz_all = sorted(glob("../gym/v5/weights/all/*.npz"))

csv_wood = (glob("../gym/v5/modelOut/wood/v1/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/wood/v2/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/wood/v3/*per_sample_val.csv"))
csv_water = (glob("../gym/v5/modelOut/water/v1/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/water/v2/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/water/v3/*per_sample_val.csv"))
csv_veg = (glob("../gym/v5/modelOut/veg/v1/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/veg/v2/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/veg/v3/*per_sample_val.csv"))
csv_all = (glob("../gym/v5/modelOut/all/v1/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/all/v2/*per_sample_val.csv")) + (glob("../gym/v5/modelOut/all/v3/*per_sample_val.csv"))

########################################
plt.figure(figsize=(16,16))
plt.subplots_adjust(wspace=0.3, hspace=0.2)

L=[]; VL=[]
for file in npz_wood:
    with np.load(file) as f:
        loss = f['loss']
        val_loss = f['val_loss']
    L.append(loss)
    VL.append(val_loss)

plt.subplot(421)
plt.semilogx(L[0], 'r-', label='Model 1, train')
plt.plot(L[1], 'g-', label='Model 2, train')
plt.plot(L[2], 'b-', label='Model 3, train')
plt.plot(VL[0], 'r--', label='Model 1, val.')
plt.plot(VL[1], 'g--', label='Model 2, val.')
plt.plot(VL[2], 'b--', label='Model 3, val.')
plt.xlabel("Training epoch")
plt.ylabel("Loss (non-dim.)")
plt.title('a)',loc='left')
plt.text(10,.6,'Wood')

L=[]; VL=[]
for file in npz_water:
    with np.load(file) as f:
        loss = f['loss']
        val_loss = f['val_loss']
    L.append(loss)
    VL.append(val_loss)

plt.subplot(423)
plt.semilogx(L[0], 'r-', label='Model 1, train')
plt.plot(L[1], 'g-', label='Model 2, train')
plt.plot(L[2], 'b-', label='Model 3, train')
plt.plot(VL[0], 'r--', label='Model 1, val.')
plt.plot(VL[1], 'g--', label='Model 2, val.')
plt.plot(VL[2], 'b--', label='Model 3, val.')
# plt.xlabel("Training epoch")
plt.ylabel("Loss (non-dim.)")
plt.title('c)',loc='left')
plt.text(10,.6,'Water')

L=[]; VL=[]
for file in npz_veg:
    with np.load(file) as f:
        loss = f['loss']
        val_loss = f['val_loss']
    L.append(loss)
    VL.append(val_loss)

plt.subplot(425)
plt.semilogx(L[0], 'r-', label='Model 1, train')
plt.plot(L[1], 'g-', label='Model 2, train')
plt.plot(L[2], 'b-', label='Model 3, train')
plt.plot(VL[0], 'r--', label='Model 1, val.')
plt.plot(VL[1], 'g--', label='Model 2, val.')
plt.plot(VL[2], 'b--', label='Model 3, val.')
# plt.xlabel("Training epoch")
plt.ylabel("Loss (non-dim.)")
plt.title('e)',loc='left')
plt.text(10,.6,'Veg')
plt.legend()

L=[]; VL=[]
for file in npz_all:
    with np.load(file) as f:
        loss = f['loss']
        val_loss = f['val_loss']
    L.append(loss)
    VL.append(val_loss)

plt.subplot(427)
plt.semilogx(L[0]/4, 'r-', label='Model 1, train')
plt.plot(L[1]/4, 'g-', label='Model 2, train')
plt.plot(L[2]/4, 'b-', label='Model 3, train')
plt.plot(VL[0]/4, 'r--', label='Model 1, val.')
plt.plot(VL[1]/4, 'g--', label='Model 2, val.')
plt.plot(VL[2]/4, 'b--', label='Model 3, val.')
# plt.xlabel("Training epoch")
plt.ylabel("Loss (non-dim.)")
plt.title('g)',loc='left')
# plt.legend()
plt.text(10,.6,'Sediment')

# plt.show()

val1_wood = [pd.read_csv(csv_wood[0]).mean()['MeanIntersectionOverUnion'],
                pd.read_csv(csv_wood[1]).mean()['MeanIntersectionOverUnion'],
                pd.read_csv(csv_wood[2]).mean()['MeanIntersectionOverUnion']]


val2_wood = [pd.read_csv(csv_wood[0]).mean()['Frequency_Weighted_Intersection_over_Union'],
                pd.read_csv(csv_wood[1]).mean()['Frequency_Weighted_Intersection_over_Union'],
                pd.read_csv(csv_wood[2]).mean()['Frequency_Weighted_Intersection_over_Union']]


val3_wood = [pd.read_csv(csv_wood[0]).mean()['MatthewsCorrelationCoefficient'],
                pd.read_csv(csv_wood[1]).mean()['MatthewsCorrelationCoefficient'],
                pd.read_csv(csv_wood[2]).mean()['MatthewsCorrelationCoefficient']]

plt.subplot(422)
plt.bar([1,2,3], [np.mean(val1_wood), np.mean(val2_wood), np.mean(val3_wood)] )
ax=plt.gca()
ax.set_xticks([1,2,3])
ax.set_xticklabels(["mIOU","fwIOU","MCC"])
plt.ylim(0,1)
plt.ylabel("Score (non-dim)")
plt.title('b)',loc='left')
plt.axhline(y=0.5, color=[.5,.5,.5], linestyle=':')

val1_water = [pd.read_csv(csv_water[0]).mean()['MeanIntersectionOverUnion'],
                pd.read_csv(csv_water[1]).mean()['MeanIntersectionOverUnion'],
                pd.read_csv(csv_water[2]).mean()['MeanIntersectionOverUnion']]


val2_water = [pd.read_csv(csv_water[0]).mean()['Frequency_Weighted_Intersection_over_Union'],
                pd.read_csv(csv_water[1]).mean()['Frequency_Weighted_Intersection_over_Union'],
                pd.read_csv(csv_water[2]).mean()['Frequency_Weighted_Intersection_over_Union']]


val3_water = [pd.read_csv(csv_water[0]).mean()['MatthewsCorrelationCoefficient'],
                pd.read_csv(csv_water[1]).mean()['MatthewsCorrelationCoefficient'],
                pd.read_csv(csv_water[2]).mean()['MatthewsCorrelationCoefficient']]

plt.subplot(424)
plt.bar([1,2,3], [np.mean(val1_water), np.mean(val2_water), np.mean(val3_water)] )
ax=plt.gca()
ax.set_xticks([1,2,3])
ax.set_xticklabels(["mIOU","fwIOU","MCC"])
plt.ylim(0,1)
plt.ylabel("Score (non-dim)")
plt.title('d)',loc='left')
plt.axhline(y=0.5, color=[.5,.5,.5], linestyle=':')

val1_veg = [pd.read_csv(csv_veg[0]).mean()['MeanIntersectionOverUnion'],
                pd.read_csv(csv_veg[1]).mean()['MeanIntersectionOverUnion'],
                pd.read_csv(csv_veg[2]).mean()['MeanIntersectionOverUnion']]


val2_veg = [pd.read_csv(csv_veg[0]).mean()['Frequency_Weighted_Intersection_over_Union'],
                pd.read_csv(csv_veg[1]).mean()['Frequency_Weighted_Intersection_over_Union'],
                pd.read_csv(csv_veg[2]).mean()['Frequency_Weighted_Intersection_over_Union']]


val3_veg = [pd.read_csv(csv_veg[0]).mean()['MatthewsCorrelationCoefficient'],
                pd.read_csv(csv_veg[1]).mean()['MatthewsCorrelationCoefficient'],
                pd.read_csv(csv_veg[2]).mean()['MatthewsCorrelationCoefficient']]

plt.subplot(426)
plt.bar([1,2,3], [np.mean(val1_veg), np.mean(val2_veg), np.mean(val3_veg)] )
ax=plt.gca()
ax.set_xticks([1,2,3])
ax.set_xticklabels(["mIOU","fwIOU","MCC"])
plt.ylim(0,1)
plt.ylabel("Score (non-dim)")
plt.title('f)',loc='left')
plt.axhline(y=0.5, color=[.5,.5,.5], linestyle=':')

val1_all = [pd.read_csv(csv_all[0]).mean()['MeanIntersectionOverUnion'],
                pd.read_csv(csv_all[1]).mean()['MeanIntersectionOverUnion'],
                pd.read_csv(csv_all[2]).mean()['MeanIntersectionOverUnion']]


val2_all = [pd.read_csv(csv_all[0]).mean()['Frequency_Weighted_Intersection_over_Union'],
                pd.read_csv(csv_all[1]).mean()['Frequency_Weighted_Intersection_over_Union'],
                pd.read_csv(csv_all[2]).mean()['Frequency_Weighted_Intersection_over_Union']]


val3_all = [pd.read_csv(csv_all[0]).mean()['MatthewsCorrelationCoefficient'],
                pd.read_csv(csv_all[1]).mean()['MatthewsCorrelationCoefficient'],
                pd.read_csv(csv_all[2]).mean()['MatthewsCorrelationCoefficient']]

plt.subplot(428)
plt.bar([1,2,3], [np.mean(val1_all), np.mean(val2_all), np.mean(val3_all)] )
ax=plt.gca()
ax.set_xticks([1,2,3])
ax.set_xticklabels(["mIOU","fwIOU","MCC"])
plt.ylim(0,1)
plt.ylabel("Score (non-dim)")
plt.title('h)',loc='left')
plt.axhline(y=0.5, color=[.5,.5,.5], linestyle=':')

# plt.show()

plt.savefig("model_skill_valset.png", dpi=300, bbox_inches="tight")
plt.close()

