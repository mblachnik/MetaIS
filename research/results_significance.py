#%%
import pandas as pd
import numpy as np
import scipy.stats as stats

df = pd.read_excel("D:\\Books\\MyPublications\\25_KBS_MetaIS\\tables\\results.xlsx",sheet_name='AUC1d',header=[0,1,2])
models = ["HMN-EI","CCIS","ENN","ICF","Drop3"]
for model in models:
    a = df.loc[:,(model,"IS","mean")].dropna().values
    b = df.loc[:, (model, "MetaIS", "mean")].dropna().values
    s,p = stats.wilcoxon(a, b, alternative= 'greater')#'two-sided')
    print(f"{model}\t{p}")