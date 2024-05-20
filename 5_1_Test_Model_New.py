import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from read_files import ImportFileToDataset, PrepareData, ImportFileToDatasetWithCommas
import os
from config.appsettings import *
import warnings
import pickle as pck
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

settings = Appsettings(RunMode.TestRun, ProcessStep.TestRun)
warnings.filterwarnings("ignore")

def metamodel_learning(names, test_set):
    
    filename = f'RF_{test_set}.sav'
    saved_model = os.path.join(settings._random_forest_models, filename)
    
    if os.path.exists(saved_model):
        rf_model = pck.load(open(saved_model, 'rb'))
        return rf_model

    meta_weight_sets = []
    
    for name in names:
        meta_weight_set = ImportFileToDataset(os.path.join(settings._fullset_meta_weights, f"{name}_meta_weights.csv"))
        meta_weight_sets.append(meta_weight_set)

    meta_weight_bigset = pd.concat(meta_weight_sets)
   
    x_train, y_train = PrepareData(meta_weight_bigset)
    
    RF_classifier = RandomForestClassifier()
    RF_classifier.fit(x_train, y_train)

   
    pck.dump(RF_classifier, open(os.path.join(saved_model), 'wb'))

    return RF_classifier
    
def instance_selection(theta, learned_classifier, test_set, i):

    meta_path = os.path.join(settings._fold_metadata, test_set, f"{test_set}-{settings.folds}-{i}tra_meta.csv")
    train_path = os.path.join(settings.original_directory, test_set, f"{settings.folds}_tra", f"{test_set}-{settings.folds}-{i}tra.csv")

    if not os.path.exists(meta_path) or not os.path.exists(train_path):
        print(f"No meta or train set for: {test_set}, {settings.folds}, iteration: {i} in instance selection")
        return pd.DataFrame(), pd.DataFrame(), 0

    meta_set = ImportFileToDataset(meta_path)
    x_meta = meta_set.copy(deep=True)

    probability = learned_classifier.predict_proba(x_meta)
    ids_over_theta = probability[:, 1] >= theta

    if(all(w == False for w in ids_over_theta)): 
        print("No probability above theta value")
        return pd.DataFrame(), pd.DataFrame(), 0

    train_set = ImportFileToDatasetWithCommas(train_path)
    train_set.index = train_set.index - 1

    x_train = train_set[ids_over_theta].iloc[:, :-1] 
    y_train = train_set[ids_over_theta].iloc[:, -1:]
    return x_train, y_train, train_set.shape[0]

def full_sets_with_weights():


    meta_weight_fullsets = [f.removesuffix(settings.meta_weights_suffix) for f in os.listdir(settings._fullset_meta_weights) if (not f[0] == '.')]
    return meta_weight_fullsets

 #------Run-----
#nauczyć na wszystkich zbiorach danych i puścić banana
#udostępnić świeży kod na gicie



datasets_names = full_sets_with_weights() 

knn_classifier = KNeighborsClassifier(n_neighbors=1)

#---Debug only
#test_sets = ["banana", "australian", "phoneme", "titanic"]
test_sets = settings.sets
except_for = ["metadata"]

result_list = [item for item in test_sets if item not in except_for]
for test_set in result_list:
#------------- 
    #---Test to skip--
    test_path = os.path.join(settings._fullset_meta_weights, f"{test_set}_meta_weights.csv")

    if not (os.path.exists(test_path)):
        print(f"No meta weights set for: {test_set}, {settings.folds}")
        continue
    #-----------------

    print(f"Start for {test_set}")
   
    paths = settings.get_paths(test_set)

    _datasets_names = [x for x in datasets_names if x != test_set]
    metamodel_classifier = metamodel_learning(_datasets_names, test_set)

    mean_accuracies = []
    mean_size_reductions = []

    for i in range(1, settings.folds+1):

        accuracies = []
        size_reductions = []
        thetas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.68, 0.7, 0.75, 0.8, 0.85]
        for index, theta in enumerate(thetas):


            print(f"i: {i}, theta: {theta}")
            x_train, y_train, size_before = instance_selection(theta, metamodel_classifier, test_set, i)
            
            if x_train.empty and y_train.empty:
                accuracies.append(-1)
                size_reductions.append(-1)
                continue

            knn_classifier.fit(x_train, y_train)

            test_path = os.path.join(paths["fold_test_set"], f"{test_set}-{settings.folds}-{i}tst.csv")
            
            test_set_load = ImportFileToDatasetWithCommas(test_path)
            test_set_load.index = test_set_load.index - 1

            x_test, y_test = PrepareData(test_set_load)

            prediction = knn_classifier.predict(x_test.values)
            accurracy = sklearn.metrics.accuracy_score(y_test, prediction)
            accuracies.append(accurracy)

            size_after = x_train.shape[0]

            size_reduction = (size_before-size_after)/size_before
            size_reductions.append(size_reduction)

            if len(mean_accuracies) < len(thetas):
                mean_accuracies.append(accurracy)
                mean_size_reductions.append(size_reduction)
            else:
                mean_accuracies[index] = mean_accuracies[index] + accurracy
                mean_size_reductions[index] = mean_size_reductions[index] + size_reduction

        result_dataframe = pd.DataFrame({"Accuracies": accuracies, "Size reduction": size_reductions, "Theta": thetas})
        result_dataframe.to_csv(os.path.join(paths["destination"], f"{test_set}_{settings.folds}-{i}.csv"), index=False)

    mean_accuracies = [x/settings.folds for x in mean_accuracies]
    mean_size_reductions = [x/settings.folds for x in mean_size_reductions]

    mean_dataframe = pd.DataFrame({"Accuracies": mean_accuracies, "Size reduction": mean_size_reductions})
    mean_dataframe.to_csv(os.path.join(paths["destination"], f"mean_{test_set}_{settings.folds}.csv"), index=False)


    
print(f"End {test_set}")



    