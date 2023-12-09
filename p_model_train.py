## Dan Buscombe, Marda Science
## 2023
import matplotlib.pyplot as plt 

import numpy as np 

from glob import glob 

wood_files = sorted(glob("../gym/v7_wood/*.npz"))

sed_files = sorted(glob("../gym/v8_sed/*.npz"))

veg_files = sorted(glob("../gym/v4_resunet/veg/*.npz"))

water_files = sorted(glob("../gym/v4_resunet/water/*.npz"))




plt.figure(figsize=(14,12))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

for counter,f in enumerate(wood_files): 
    with np.load(f, allow_pickle=True) as dat:
        data = dict()
        for k in dat.keys():
            data[k] = dat[k]
        del dat

    plt.subplot(421)
    if counter==0:
        plt.plot(data['loss'],'-',color=[.75,.75,.75])
        plt.plot(data['val_loss'],'--',color=[.75,.75,.75])
        plt.ylabel("Loss (-)"); plt.xlabel("Epoch (-)")
        plt.title('a) Wood', loc='left'); 

    if counter==1:
        plt.plot(data['loss'],'-',color='m')
        plt.plot(data['val_loss'],'--',color='m')
    if counter==2:
        plt.plot(data['loss'],'-',color='g')
        plt.plot(data['val_loss'],'--',color='g')
    plt.xlim(0,80); plt.ylim(0,1.75)

    plt.subplot(422)
    if counter==0:
        plt.plot(data['lr'],'-',color=[.75,.75,.75])
        plt.ylabel("Learning rate (-)"); plt.xlabel("Epoch (-)")
    if counter==1:
        plt.plot(data['lr'],'-',color='m')
    if counter==2:
        plt.plot(data['lr'],'-',color='g')
        plt.ylim(0,80); plt.xlim(0,0.0001)
    plt.xlim(0,80); plt.ylim(0,0.0001)


for counter,f in enumerate(sed_files): 
    with np.load(f, allow_pickle=True) as dat:
        data = dict()
        for k in dat.keys():
            data[k] = dat[k]
        del dat

    plt.subplot(423)
    if counter==0:
        plt.plot(data['loss'],'-',color=[.75,.75,.75])
        plt.plot(data['val_loss'],'--',color=[.75,.75,.75])
        plt.ylabel("Loss (-)"); plt.xlabel("Epoch (-)")
        plt.title('b) Sediment', loc='left'); 

    if counter==1:
        plt.plot(data['loss'],'-',color='m')
        plt.plot(data['val_loss'],'--',color='m')
    if counter==2:
        plt.plot(data['loss'],'-',color='g')
        plt.plot(data['val_loss'],'--',color='g')
    plt.xlim(0,80); plt.ylim(0,1.75)

    plt.subplot(424)
    if counter==0:
        plt.plot(data['lr'],'-',color=[.75,.75,.75])
        plt.ylabel("Learning rate (-)"); plt.xlabel("Epoch (-)")
    if counter==1:
        plt.plot(data['lr'],'-',color='m')
    if counter==2:
        plt.plot(data['lr'],'-',color='g')
        plt.ylim(0,80); plt.xlim(0,0.0001)
    plt.xlim(0,80); plt.ylim(0,0.0001)


for counter,f in enumerate(veg_files): 
    with np.load(f, allow_pickle=True) as dat:
        data = dict()
        for k in dat.keys():
            data[k] = dat[k]
        del dat

    plt.subplot(425)
    if counter==0:
        plt.plot(data['loss'],'-',color=[.75,.75,.75])
        plt.plot(data['val_loss'],'--',color=[.75,.75,.75])
        plt.ylabel("Loss (-)"); plt.xlabel("Epoch (-)")
        plt.title('c) Vegetation', loc='left'); 
    if counter==1:
        plt.plot(data['loss'],'-',color='m')
        plt.plot(data['val_loss'],'--',color='m')
    if counter==2:
        plt.plot(data['loss'],'-',color='g')
        plt.plot(data['val_loss'],'--',color='g')
    plt.xlim(0,80); plt.ylim(0,1.75)

    plt.subplot(426)
    if counter==0:
        plt.plot(data['lr'],'-',color=[.75,.75,.75])
        plt.ylabel("Learning rate (-)"); plt.xlabel("Epoch (-)")
    if counter==1:
        plt.plot(data['lr'],'-',color='m')
    if counter==2:
        plt.plot(data['lr'],'-',color='g')
    plt.xlim(0,80); plt.ylim(0,0.0001)


for counter,f in enumerate(water_files): 
    with np.load(f, allow_pickle=True) as dat:
        data = dict()
        for k in dat.keys():
            data[k] = dat[k]
        del dat

    plt.subplot(427)
    if counter==0:
        plt.plot(data['loss'],'-',color=[.75,.75,.75])
        plt.plot(data['val_loss'],'--',color=[.75,.75,.75])
        plt.ylabel("Loss (-)"); plt.xlabel("Epoch (-)")
        plt.title('d) Water', loc='left'); 
    if counter==1:
        plt.plot(data['loss'],'-',color='m')
        plt.plot(data['val_loss'],'--',color='m')
    if counter==2:
        plt.plot(data['loss'],'-',color='g')
        plt.plot(data['val_loss'],'--',color='g')
    plt.xlim(0,80); plt.ylim(0,1.75)

    plt.subplot(428)
    if counter==0:
        plt.plot(data['lr'],'-',color=[.75,.75,.75])
        plt.ylabel("Learning rate (-)"); plt.xlabel("Epoch (-)")
    if counter==1:
        plt.plot(data['lr'],'-',color='m')
    if counter==2:
        plt.plot(data['lr'],'-',color='g')
    plt.xlim(0,80); plt.ylim(0,0.0001)


# plt.show()
plt.savefig("summaries/all_models_train_curves.png", dpi=300, bbox_inches="tight")

plt.close()
