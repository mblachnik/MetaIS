import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,matthews_corrcoef
def read_data(fName):
    df_train = pd.read_csv(fName, sep=";")
    X = df_train.loc[:, [c for c in df_train.columns if c not in ["LABEL", "id"]]]#.values
    y = df_train.loc[:, "LABEL"]#.values
    return X,y

def score(yp,y):
    res = {"acc": accuracy_score(y_true=y,y_pred=yp),
            "f1": f1_score(y_true=y,y_pred=yp, average='micro'),
            'bacc': balanced_accuracy_score(y_true=y,y_pred=yp),
            'mcc':matthews_corrcoef(y_true=y,y_pred=yp),
           }
    return res