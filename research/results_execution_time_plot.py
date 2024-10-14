import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
data_dir = "Y:\\MetaIS\\results\\results_postprocessing\\"
dat_dsc = pd.read_csv(os.path.join(data_dir,"datasets_2header.csv"),sep=",",header=[0,1]).set_index("Dataset",drop=True)
dat_dsc = dat_dsc[[col for col in dat_dsc.columns if col[0]!="Id"]]
dat_dsc.index = [idx[0] for idx in dat_dsc.index]
#%%
dat_time = pd.read_csv(os.path.join(data_dir,"summary_of_execution_time.csv"),sep=",",header=[0,1]).set_index("Dataset",drop=True)
dat_time.index = [idx[0] for idx in dat_time.index]
#%%
dat = dat_dsc.join(dat_time)

models = ["CCIS"]#,"HMEI","ENN","ICF","Drop3"]
plt.figure(1,clear=True)
dat = dat.sort_values(by=dat.columns[0])
for model in models:
    y1 = dat.loc[:,(model,"process_time_IS")]
    y1=y1/1000 #Zamiana z ms na s
    y2 = dat.loc[:,(model,"process_time_meta_IS")]
    x= dat.loc[:,"samples"]
    plt.plot(x,y1,"-x")
    plt.plot(x, y2,"-x")
