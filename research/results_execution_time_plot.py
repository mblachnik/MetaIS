"""
Script to plot execution time comparison between methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l
#%%
data_dir = "data/results/results_postprocessing/"#"Y:\\MetaIS\\results\\results_postprocessing\\"

dat_dsc = pd.read_csv(os.path.join(data_dir,"datasets_2header.csv"),sep=",",header=[0,1]).set_index("Dataset",drop=True)
dat_dsc = dat_dsc[[col for col in dat_dsc.columns if col[0]!="Id"]]
dat_dsc.index = [idx[0] for idx in dat_dsc.index]
#%%
dat_time = pd.read_csv(os.path.join(data_dir,"summary_of_execution_time.csv"),sep=",",header=[0,1]).set_index("Dataset",drop=True)
dat_time.index = [idx[0] for idx in dat_time.index]
#%%
dat = dat_dsc.join(dat_time)

models = ["CCIS","HMEI","ENN","ICFKeel","Drop3Keel"]
dat = dat.sort_values(by=dat.columns[0])
col=['r','g','b','c','m','k']
#dat = dat.dropna()
#%%
plt.figure(1,clear=True)
for i,model in enumerate(models):
    Y1 = dat.loc[:, (model, "process_time_IS")]
    #Y1= Y1 / 1000 #Zamiana z ms na s
    Y2 = dat.loc[:, (model, "process_time_meta_IS")]
    X= dat.loc[:, "samples"]
    plt.plot(X, Y1, "-x", color=col[i], label=model)
    plt.plot(X, Y2, ":x", color=col[i])

b0=-30
b = b0
mi_b = b
xo=-10000
line_width = 15
ids = []
for j,x in enumerate(X.values.reshape((-1,))):

    if x-xo < 5000:
        b -= line_width
    else:
        b = b0
    print(f"{xo} | {x} | {x - xo} | {b} {dat.index[j]}")

    ids.append((x,b,dat.index[j]))
    if b<mi_b:
        mi_b = b
    xo = x
ymax=300
for x,b,s in ids:
    b = (mi_b+b0) - b
    plt.text(x, b, s)
    #newline((x,b),(x,ymax))
    plt.plot([x,x],[b+line_width,ymax],':k')
plt.plot([0,600_000],[0,0],'k',linewidth=1)
plt.ylim(mi_b-line_width,ymax)
plt.xlim(2000,60_000)
plt.legend()