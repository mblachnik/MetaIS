#%%
from http.cookiejar import deepvalues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = "Y:\\MetaIS\\results\\"

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

# def savePlot(i, dir, model):
#     postfix = "_2"
#     plt.figure(i,clear=True)
#     me_norm.plot.bar()
#     plt.tight_layout()
#     plt.gcf().savefig(f"{dir}figs\\fig_feat_imp_k_detailed_{model}{postfix}.png")

def savePlot(i, name, plot):
    plt.figure(i, clear=True, figsize=(10, 6))  # większy rozmiar wykresu
    ax = plot.plot.bar(color="skyblue", edgecolor="black")  # lepsze kolory i obramowanie

    # Tytuł i etykiety osi
    # ax.set_title("Feature Importance (normalized)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Features", fontsize=16)
    ax.set_ylabel("Importance", fontsize=16)

    # Poprawa czytelności etykiet
    plt.xticks(rotation=45, ha="right", fontsize=14)  
    plt.yticks(fontsize=14)

    # Siatka dla łatwiejszego odczytu
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.gcf().savefig(f"{name}.png", dpi=300)

for i,model in enumerate(models):
    cur_dir = dir + model + "\\"
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
    savePlot(i, f"{dir}figs/fig_feat_imp_k_detailed_{model}", me_norm)

    res = {}
    for group in rows_core+row_groups:
        rows = [row for row in me.index if group in row]
        res[group] = me.loc[rows].sum()
    group_imp = pd.Series(res)
    group_imp.index = [row_names_dict[row] for row in group_imp.index]

    savePlot(10+i, f"{dir}figs\\fig_feat_imp_type_{model}", group_imp)

    res_k = {row:me.loc[row]for row in rows_core}
    for group in kGroups:
        rows = [row for row in me.index if group in row]
        res_k[group] = me.loc[rows].sum()
    group_imp_k = pd.Series(res_k)
    group_imp_k.index = [kGroup_names_dict[row] for row in group_imp_k.index]

    savePlot(20+i, f"{dir}figs\\fig_feat_imp_k_{model}", group_imp_k)
