"""
Drow relation between prediction performance and compression. The plot draws it per model for different datasets.
On a single plot we have same model but different datasets.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.colors as colors

from research.basics.utils import loadConfig, savePlotFig

font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

doSave = False
config = loadConfig()


#%%
results_files = ["results_MetaIS_agg_v0_large.csv",
                 #"results_MetaIS_agg_v6.csv"
#"results_MetaIS_agg_v_xxx.csv",
#"results_MetaIS_agg_v_xxx1.csv"
               ]
for model in config["models"]:

    dfs = []
    for file in results_files:
        df = pd.read_csv(os.path.join(config["results_dir"] + model,file),header=[0,1])
        dfs.append(df)
    df = pd.concat(dfs,axis=0)

    df_ref = pd.read_csv(os.path.join(config["results_dir"]+model,"results_IS_agg_v0.csv"),header=[0,1])


    #%%
    datasets = np.unique(df['name'])
    cols = colors.get_named_colors_mapping()
    cols = ['r','g','b']
    plots_per_image = 3
    for i,ds in enumerate(datasets):
        print(f"{i}) {ds}")
        col = i % plots_per_image
        df_tmp = df[(df['name']==ds).values]
        df_ref_tmp = df_ref[(df_ref['name']==ds).values]
        if i % plots_per_image==0:
            plt.figure(i,clear=True)
        plt.plot(df_tmp[('red_rate','mean')],df_tmp[('f1','mean')],color=cols[col],label=ds,marker='*')
        plt.plot(df_ref_tmp[('red_rate', 'mean')], df_ref_tmp[('f1', 'mean')],color=cols[col],marker='x', markersize=13)
        plt.xlabel("Reduction rate [-]",fontsize=font['size'])
        plt.ylabel("F1 [-]",fontsize=font['size'])
        if i % plots_per_image==plots_per_image-1:
            plt.legend()
            if doSave:
                savePlotFig(config, model, f"fig{i}.png")
            plt.show()

    plt.legend()
    if doSave:
        savePlotFig(config, model, f"fig{i}.png")
    #plt.show()


