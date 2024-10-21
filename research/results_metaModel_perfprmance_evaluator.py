"""
This script is to evaluate the performance of the metamodel during training. Or how different models influence performance of the metoa model.

This experiments are based on the CCIS meta_features
"""
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
directory = "data\\meta_model_train_data\\"
X = pd.read_csv(directory + "X.csv",sep=",")
y = pd.read_csv(directory + "y.csv",sep=",")
X = X.iloc[:,1:-1]
y = y.iloc[:,1]

models = [#RandomForestClassifier(n_estimators=300,n_jobs=4),
          #BalancedRandomForestClassifier(n_estimators=300,n_jobs=4),
          #RandomForestClassifier(n_jobs=4),
          BalancedRandomForestClassifier(n_jobs=4)
          ]
res = []
for i,model in enumerate(models):
    kfold = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
    sc = cross_val_score(estimator=model,X=X,y=y,cv=kfold,n_jobs=4,scoring='balanced_accuracy')
    res.append((np.mean(sc),np.std(sc)))
    print(f"{np.mean(sc)}\t{np.std(sc)}")
##model = RandomForestClassifier(def)         0.9149487250622841	0.0011635128597316123
#model = BalancedRandomForestClassifier(n_estimators=300,n_jobs=4) 0.8370874021998695	0.001480023981170922
#model = RandomForestClassifier(n_estimators=300,n_jobs=4), 0.9156560048343607	0.0010772285907380353
#model = BalancedRandomForestClassifier(n_jobs=4)

