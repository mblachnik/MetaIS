"""
Script to run experiments in which we train meta model based
on already extracted metafeatures.
In order to get meatfeatures run exp1_generate_meta.py
"""

#%%
from joblib import Parallel, delayed
import numpy as np
from sklearn.base import clone
from instance_selection.metais_tools import loadMetaFromDatasets, train, store, generateMetaForDatasets
import os
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier

from research.basics.utils import loadConfig

#meta_model = RandomForestClassifier(n_jobs=5)
meta_model = BalancedRandomForestClassifier(n_jobs=5)

doGenerateMeta = False #Jak true to generujemy metaatrybuty, jak False to pomijamy ten krok

config = loadConfig()

def trainMeta(fNames, Xs, ys, fN, config,is_model,meta_model):
    Xx = []
    yy = []
    for Xt,yt,ft in zip(Xs,ys,fNames):
        if ft == fN: continue
        Xx.append(Xt)
        yy.append(yt)

    X = np.vstack(Xx) #Sklejenie części X
    y = np.hstack(yy) #Sklejenie części y
    model = train(X,y,model=meta_model) #Nauczenie meta modelu
    models_dir = config["models_dir"] + is_model + "/"
    store(model, file = f"{models_dir}/model_{fN}.pickl") #Zapisanie meta_modelu

#Tworzy listę toupli składającą się z katalogu, nazwy pliku i rozszerzenia przechodząc po podkatalogach katalogu data
#Spośród wybranych plików wybierane są te które zawierają  _meta i są ty[u .csv
for model in config["models"]:
    files = [(r, f.replace(".csv",""), ".csv")
             for r,ds,fs in os.walk(config["data_dir"]+model+"/")
                for f in fs
                    if f.endswith(".csv")
                        and ("_meta" in f)
                        and (not ("tra." in f)) and (not ("tst." in f)) #We train metamodel only for full datasets without split into training and testing
                        and any( True  if c in f else False for c in config['datasets'])
             ]
    #%%
    #Ładowanie plików i uruchomienie procesu uczenia meta-modelu
    Xs, ys, fNames = loadMetaFromDatasets(files)
    n_jobs = config["n_jobs"]
    #%%
    #The above function gets a list of X and y of metaattributes for given file, this allows us to combine only selected files and train metamodel only on selected files


    if n_jobs > 1: #We iterate over files so that for each file a separate dataset of metaattributes will be created, but from the full list of Xs and ys we have to droop current file becouse for that file we will perform experiments.
        Parallel(n_jobs=n_jobs, prefer="threads", backend="loky")(delayed(trainMeta)(fNames, Xs, ys, fN, config, model, clone(meta_model)) for fN in tqdm(fNames))
    else:
        for fN in tqdm(fNames):
            trainMeta(fNames, Xs, ys, fN, config, model, clone(meta_model))
