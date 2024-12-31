import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from research.basics.utils import getResultsFilePathWithPostfix, loadConfig, savePlotFig
#%%

config = loadConfig()

di = config['results_dir']
models = config["models"]

font = { 'size'   : 12}

matplotlib.rc('font', **font)

ref_model = "CCIS"
file_path = getResultsFilePathWithPostfix(config, ref_model, True, False, config["IS_result_postfix"][0])
df_CCIS = pd.read_csv(file_path,header=[0,1])
plots_per_image = 3

file_postfix = config["result_postfix"][0]
save = True

file = f"results_MetaIS_{file_postfix}.csv"

df = {}
cols = ['r','g','b']

models_dict = {}

for model in models:
    file_path = getResultsFilePathWithPostfix(config, model, True, True, file_postfix)
    temp = pd.read_csv(file_path,header=[0,1])
    models_dict[model] = temp

df = next(iter(models_dict.values()))
datasets = np.unique(df_CCIS['name'])
cols = ['r','g','b']

for i, model in enumerate(models):
    plt.figure(i,clear=True)
    label_datasets = ""
    plot_no = 0
    for j,ds in enumerate(datasets):
        df = models_dict[model]
        df_tmp = df[(df['name']==ds).values]
        if df_tmp.empty:
            continue
        df_ref_tmp = df_CCIS[(df_CCIS['name']==ds).values]

        if plot_no == plots_per_image:
            plt.xlabel("Reduction rate [-]",fontsize=font['size'])
            plt.ylabel("F1 [-]",fontsize=font['size'])
            plt.legend()
            if(save):
                savePlotFig(config, model, f"{file_postfix}_vs_{ref_model}_{label_datasets}")
            else:
                plt.show()
            plt.figure(i,clear=True)
            label_datasets = ""
            plot_no = 0
        
        label_datasets += f"_{ds}"
        color = cols[plot_no]
        plot_no += 1

        plt.plot(df_tmp[('red_rate','mean')],df_tmp[('f1','mean')],color=color,label=ds,marker='*')
        plt.plot(df_ref_tmp[('red_rate', 'mean')], df_ref_tmp[('f1', 'mean')],color=color,marker='x', markersize=13,linewidth=5)
    plt.xlabel("Reduction rate [-]",fontsize=font['size'])
    plt.ylabel("F1 [-]",fontsize=font['size'])
    plt.legend()
    if(save):
        savePlotFig(config, model, f"{file_postfix}_vs_{ref_model}{label_datasets}")
    else:
        plt.show()

    