import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from research.basics.utils import loadConfig, savePlotFig

font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

doSave = True
doShow = False
config = loadConfig()


#%%
results_files = ["results_MetaIS_agg_v12_large.csv",
                 #"results_MetaIS_agg_v6.csv"
#"results_MetaIS_agg_v_xxx.csv",
#"results_MetaIS_agg_v_xxx1.csv"
               ]

ref_result_file = "results_IS_agg_v5.csv"


models_dict = {}
models_ref_dict = {}
for model in config["models"]:
    dfs = []
    for file in results_files:
        temp = pd.read_csv(os.path.join(config["results_dir"] + model,file),header=[0,1])
        dfs.append(temp)
    models_dict[model] = pd.concat(dfs,axis=0)
    models_ref_dict[model] = pd.read_csv(os.path.join(config["results_dir"]+model,ref_result_file),header=[0,1])



#%%
df = next(iter(models_dict.values()))
datasets = np.unique(df['name'])
cols = ['r','g','b', 'y', 'k','m']
for j,ds in enumerate(datasets):
    plt.figure(j,clear=True)
    for i, model in enumerate(config["models"]):
        df_ref = models_ref_dict[model]
        df = models_dict[model]
        df_tmp = df[(df['name']==ds).values]
        df_ref_tmp = df_ref[(df_ref['name']==ds).values]
        model_name = config["models_names"][model]
        plt.plot(df_tmp[('red_rate','mean')],df_tmp[('f1','mean')],color=cols[i],label=model_name,marker='*')
        plt.plot(df_ref_tmp[('red_rate', 'mean')], df_ref_tmp[('f1', 'mean')],color=cols[i],marker='x', markersize=13,linewidth=5)
    plt.xlabel("Reduction rate [-]",fontsize=font['size'])
    plt.ylabel("F1 [-]",fontsize=font['size'])
    plt.legend()
    if doSave:
        savePlotFig(config, "", f"fig_{ds}.png")
    if doShow:
        plt.show()


