import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
di = "D:\\Desktop\\tmp\\"
models = ["CCIS",
          "HMEI_ENN_ICFKeel_CCIS",
          "HMEI_ICFKeel_CCIS"]
fName = "results_MetaIS_v13_large_php89ntbG-5-5tra.dat.csv"

ress = {model:pd.read_csv(os.path.join(di,model,fName),sep=",") for model in models}

plt.figure(1,clear=True)
for model,df in ress.items():
    x = df["red_rate"]
    y = df['acc']
    plt.plot(x,y,label=model)
plt.legend()