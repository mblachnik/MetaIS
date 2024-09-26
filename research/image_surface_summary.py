import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
from scipy.interpolate import interp1d

from research.basics.utils import loadConfig, savePlotFig

config = loadConfig()

model = "HMEI"
df_1nn = pd.read_csv(os.path.join(config["results_dir"],'1NN',"results_IS_agg_v0.csv"),header=[0,1])
df_meta = pd.read_csv(os.path.join(config["results_dir"],model,"results_MetaIS_agg_v12_large.csv"),header=[0,1])
df = pd.concat([df_1nn, df_meta],axis=0)

def_hmei = pd.read_csv(os.path.join(config["results_dir"],model,"results_IS_agg_v5.csv"),header=[0,1])
df_ref = pd.concat([df_1nn, def_hmei],axis=0)

plots_per_image = 1
ds = 'ring'
df_tmp = df[(df['name']==ds).values]
df_ref_tmp = df_ref[(df_ref['name']==ds).values]

X = df_tmp[('red_rate','mean')]
Y = df_tmp[('acc','mean')]

X_REF = df_ref_tmp[('red_rate', 'mean')]
Y_REF = df_ref_tmp[('acc', 'mean')]

meta_color = 'r'
ref_color = 'g'

plt.plot(X,Y,color=meta_color,label=ds,marker='*')
plt.plot(X_REF, Y_REF,color=ref_color,marker='o')

plt.xlabel('Compression [%]')
plt.ylabel('Accuracy [%]') 

plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)

plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['left'].set_color('none')
plt.gca().spines['bottom'].set_color('none')

plt.fill_between(X, Y, color='white', edgecolor=meta_color, hatch='//', alpha=0.5)
plt.fill_between(X_REF, Y_REF, color='white', edgecolor=ref_color, hatch='\\\\',alpha=0.5)

figs_dir = os.path.join(f"{config['results_dir']}", "figure1.png")
plt.savefig(figs_dir)
plt.show()

plt.plot(X,Y,color=meta_color,label=ds,marker='*')
plt.plot(X_REF, Y_REF,color=ref_color,marker='o')

plt.xlabel('Compression [%]')
plt.ylabel('Accuracy [%]') 

plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)

plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['left'].set_color('none')
plt.gca().spines['bottom'].set_color('none')

interp_func = interp1d(X, Y, kind='linear', fill_value='extrapolate')
x_target = X_REF.iloc[1]
y_target = interp_func(x_target)
X_filtered = X.clip(upper=x_target)
Y_filtered = Y.head(len(X_filtered))
X_filtered.iloc[-1] = x_target
Y_filtered.iloc[-1] = y_target

plt.fill_between(X_filtered, Y_filtered, color='white', edgecolor=meta_color, hatch='//', alpha=0.5)
plt.fill_between(X_REF, Y_REF, color='white', edgecolor=ref_color, hatch='\\\\',alpha=0.5)
figs_dir = os.path.join(f"{config['results_dir']}", "figure2.png")
plt.savefig(figs_dir)
plt.show()


