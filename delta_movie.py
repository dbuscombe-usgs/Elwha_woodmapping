

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.io import imread 
from glob import glob 
import os 
import matplotlib.ticker as ticker
from tkinter import filedialog
from tkinter import *
import matplotlib as mpl

def make_ani_sidebyside(files1, files2):

    cmap = mpl.colors.ListedColormap(["lawngreen", "blue", "gold", "brown"])

    fig, ax = plt.subplots(1,2)
    ims = []
    for f1, f2 in zip(files1, files2):
        im1 = ax[0].imshow(imread(f1), animated=True)
        ax[0].xaxis.set_major_locator(ticker.NullLocator())
        ax[0].yaxis.set_major_locator(ticker.NullLocator())
        t1=ax[0].text(180,20,f1.split(os.sep)[-1].split('_')[0], color='w', animated=True)

        im2 = ax[1].imshow(imread(f2), animated=True, cmap=cmap, vmin=0, vmax=3)
        ax[1].xaxis.set_major_locator(ticker.NullLocator())
        ax[1].yaxis.set_major_locator(ticker.NullLocator())
        t2=ax[1].text(180,20,f2.split(os.sep)[-1].split('_')[0], color='w', animated=True)

        ims.append([im1,t1, im2, t2])
        # ax.clear()

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=0)
    return ani


#====================================================================
## user navigate to folder of files
root = Tk()
root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory of image files")
folder = root.filename
print(folder)
root.withdraw()

files = sorted(glob(folder+os.sep+"*.jpg"))

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory of label files")
folder = root.filename
print(folder)
root.withdraw()

label_files = sorted(glob(folder+os.sep+"*.jpg"))

fps = 1
#====================================================================
ani = make_ani_sidebyside(files,label_files)
ani.save(folder+os.sep+"out_sidebyside.gif", writer='imagemagick', fps=fps)
del ani

## to crop ...
## convert out_sidebyside.gif -delay 100 -loop 0 -coalesce -repage 0x0 -crop 510x238+73+124 +repage out_sidebyside_cropped_100ms.gif