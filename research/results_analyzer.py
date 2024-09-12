import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import matplotlib.colors as colors

doSave = True
config_file = "config.yaml"
if not os.path.isfile(config_file):
    config_file = "../config.yaml"
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)


#%%
results_files = ["results_MetaIS_agg_v11_large.csv",
                 #"results_MetaIS_agg_v6.csv"
#"results_MetaIS_agg_v_xxx.csv",
#"results_MetaIS_agg_v_xxx1.csv"
               ]

model = "Drop3Keel"
dfs = []
for file in results_files:
    df = pd.read_csv(os.path.join(config["results_dir"] + model,file),header=[0,1])
    dfs.append(df)
df = pd.concat(dfs,axis=0)

df_ref = pd.read_csv(os.path.join(config["results_dir"]+model,"results_IS_agg_v5.csv"),header=[0,1])


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
    plt.plot(df_tmp[('red_rate','mean')],df_tmp[('acc','mean')],color=cols[col],label=ds,marker='*')
    plt.plot(df_ref_tmp[('red_rate', 'mean')], df_ref_tmp[('acc', 'mean')],color=cols[col],marker='o')
    if i % plots_per_image==plots_per_image-1:
        plt.legend()
        if doSave:
            if not os.path.isdir(os.path.join(config['results_dir']+model,"figs")):
                os.mkdir(os.path.join(config['results_dir']+model,"figs"))
            plt.savefig(os.path.join(config['results_dir']+model,"figs", f"fig{i}.png"))
        plt.show()

plt.legend()
if doSave:
    if not os.path.isdir(os.path.join(config['results_dir']+model,"figs")):
        os.mkdir(os.path.join(config['results_dir']+model,"figs"))
    plt.savefig(os.path.join(config['results_dir']+model,"figs", f"fig{i}.png"))
plt.show()


