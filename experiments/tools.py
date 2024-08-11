"""
Some simple methods to simplify running the experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,matthews_corrcoef
def read_data(fName):
    df_train = pd.read_csv(fName, sep=";")
    X = df_train.loc[:, [c for c in df_train.columns if c not in ["LABEL", "id"]]]#.values
    y = df_train.loc[:, "LABEL"]#.values
    return X,y

def read_data_and_IS(fName, fNameIS):
    df_tr = pd.read_csv(fName, sep=";")
    df_is = pd.read_csv(fNameIS, sep=";")
    df_is.rename(columns={"weight": "_weight_"}, inplace=True)
    df = pd.merge(df_tr, df_is, on='id', how='outer')
    if df.isnull().any().any():
        raise ValueError(f"In {fName} after mergeing with IS weights one of values is NAN but it shouldnt. \n"
                         f"It is likely that {fName} and {fNameIS} do not match")
    idx = df["_weight_"] == 1

    X = df.loc[idx, [c for c in df.columns if c not in ["LABEL", "id","_weight_"]]]#.values
    y = df.loc[idx, "LABEL"]#.values

    return X,y,scoreIS(df_tr,X)

def score(yp,y):
    res = {"acc": accuracy_score(y_true=y,y_pred=yp),
            "f1": f1_score(y_true=y,y_pred=yp, average='micro'),
            'bacc': balanced_accuracy_score(y_true=y,y_pred=yp),
            'mcc':matthews_corrcoef(y_true=y,y_pred=yp),
           }
    return res

def scoreIS(X_tr, X_sel):
    stats = {"comp": X_sel.shape[0] / X_tr.shape[0],
             "red_rate": (X_tr.shape[0] - X_sel.shape[0]) / X_tr.shape[0]
             }
    return stats


def hist_classification(df:pd.DataFrame,col_x, col_y, bins=10):
    _, edges = np.histogram(df[col_x], bins=bins)
    histdata = []
    labels = []
    for n, group in df.groupby(col_y):
        histdata.append(np.histogram(group[col_x], bins=edges)[0])
        labels.append(n)

    hist = np.array(histdata)
    histcum = np.cumsum(hist, axis=0)

    plt.bar(edges[:-1], hist[0, :], width=np.diff(edges)[0],
            label=labels[0], align="edge")

    for i in range(1, len(hist)):
        plt.bar(edges[:-1], hist[i, :], width=np.diff(edges)[0],
                bottom=histcum[i - 1, :], label=labels[i], align="edge")
    plt.legend(title="class")