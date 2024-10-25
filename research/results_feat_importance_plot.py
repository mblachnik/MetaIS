#%%
from http.cookiejar import deepvalues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = "D:\\Projects\\DataMining\\scripts\\CI\\2024_MetaIS\\" + "data\\results\\"

models = ["CCIS","HMEI","ICFKeel","Drop3Keel","ENN"]

#%%
rows_core = ["minDistanceSameClass", "minDistanceOppositeClass", "minDistanceAnyClass"]
rows_core_names = ["Min. Dist. Same Class", "Min. Dist. Oppo. Class", "Min. Dist. Any Class"]

row_groups = ["sameClassNeighbors", "oppositeClassNeighbors", "meanDistanceAnyClass", "meanDistanceSameClass", "meanDistanceOppositeClass"]
row_groups_names = ["Same Class Neighbors", "Oppo. Class Neighbors", "Mean Dist. any Class", "Mean Dist. Same Class", "Mean Dist. Oppo. Class"]

row_names_dict = dict(zip(rows_core + row_groups, rows_core_names + row_groups_names))

#kGroups = ["3_5","9_15","23_33"]
#kGroups_names = [" (3,5)"," (9,15)"," (23,33)"]

kGroups = ["3","5","9","15","23","33"]
kGroups_names = ["k=3","k=5","k=9","k=15","k=23","k=33"]

kGroup_names_dict = dict(zip(kGroups+rows_core,kGroups_names+rows_core_names))


for i,model in enumerate(models):
    cur_dir = dir + model + "\\Feat_import\\"
    df = pd.read_csv(cur_dir+"feat_imp_summary4.csv")
    df = df.set_index("Meta attribute", drop=True)
    df.columns = df.iloc[0,:]

    rows = [row for row in rows_core]
    for k in kGroups:
        rows_t = [col + k for col in row_groups]
        rows.extend(rows_t)
    df = df.loc[rows,:]
    me = df.mean(axis=1)
    me_norm = me.copy(deep=True)#/me.sum()
    new_names = []

    for row in me_norm.index:
        n1 = [n for n in (rows_core+row_groups) if n in row]
        n2 = [n for n in kGroups if n in row]
        row = row.replace(row, row_names_dict[n1[0]])
        if len(n2):
            row += kGroup_names_dict[n2[-1]]
        new_names.append(row)
    me_norm.index = new_names
    plt.figure(i,clear=True)
    me_norm.plot.bar()
    plt.tight_layout()
    plt.gcf().savefig(f"{dir}figs\\fig_feat_imp_k_detailed_{model}.png")

    res = {}
    for group in rows_core+row_groups:
        rows = [row for row in me.index if group in row]
        res[group] = me.loc[rows].sum()
    group_imp = pd.Series(res)
    group_imp.index = [row_names_dict[row] for row in group_imp.index]

    plt.figure(10+i,clear=True)
    group_imp.plot.bar()
    plt.tight_layout()
    plt.gcf().savefig(f"{dir}figs\\fig_feat_imp_type_{model}.png")



    res_k = {row:me.loc[row]for row in rows_core}
    for group in kGroups:
        rows = [row for row in me.index if group in row]
        res_k[group] = me.loc[rows].sum()
    group_imp_k = pd.Series(res_k)
    group_imp_k.index = [kGroup_names_dict[row] for row in group_imp_k.index]

    plt.figure(20+i,clear=True)
    group_imp_k.plot.bar()
    plt.tight_layout()
    plt.gcf().savefig(f"{dir}figs\\fig_feat_imp_k_{model}.png")
