import os
import warnings
import pandas as pd
import sklearn
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from config.appsettings import Appsettings
from read_files import ImportFileToDatasetWithCommas, PrepareData
from type_dictionaries import ProcessStep, RunMode

def GetDestinationPath(settings: Appsettings):
    path = os.path.join(settings.weights_directory, "Standard_Results")

    if not os.path.exists(path):
        os.makedirs(path)
    return path

def ImportFileToDataset(path):
    dataset = pd.read_csv( 
            filepath_or_buffer=path, 
            sep=";", 
            encoding="utf-8",
            ##header=0,
            ##index_col=0,
            decimal=".") 
    return dataset

settings = Appsettings(RunMode.TestRun, ProcessStep.TestRun)
warnings.filterwarnings("ignore")

knn_classifier = KNeighborsClassifier(n_neighbors=1)

calculated_sets = []
mean_accuracies = []
mean_size_reductions = []

result_path = GetDestinationPath(settings)

test_sets = settings.sets
for test_set in test_sets:

    print(f"Start set {test_set}")
    paths = settings.get_paths(test_set)

    accuracies = []
    size_reductions = []

    for i in range(1, settings.folds+1):

        #Load paths
        tra_path = os.path.join(settings.original_directory, test_set, f"{settings.folds}_tra", f"{test_set}-{settings.folds}-{i}tra.csv")
        proto_path = os.path.join(settings.weights_directory, test_set, f"{settings.folds}_tra_proto", f"{test_set}-{settings.folds}-{i}tra.dat_proto.csv")
        test_path = os.path.join(paths["fold_test_set"], f"{test_set}-{settings.folds}-{i}tst.csv")

        if not os.path.exists(tra_path) or not os.path.exists(proto_path) or not os.path.exists(test_path):
            print(f"One of paths for set {test_set} doesn't exist")
            continue

        #Load sets
        tra_set = ImportFileToDatasetWithCommas(tra_path)
        proto_set = ImportFileToDataset(proto_path)
  
        #Filtered training set by algorithn
        filtered_set = tra_set[tra_set.index.isin(proto_set['id'])]

        x_filtered_set = filtered_set.iloc[:, :-1] 
        y_filtered_set = filtered_set.iloc[:, -1:]

        #Learn 1NN classifier
        knn_classifier.fit(x_filtered_set, y_filtered_set)

        #Load test
        test_set_load = ImportFileToDatasetWithCommas(test_path)
        test_set_load.index = test_set_load.index - 1

        x_test, y_test = PrepareData(test_set_load)

        #Predict
        prediction = knn_classifier.predict(x_test.values)

        #Accuracy
        accurracy = sklearn.metrics.accuracy_score(y_test, prediction)

        #Size reduction
        size_before = tra_set.shape[0]
        size_after = proto_set.shape[0]
        size_reduction = (size_before-size_after)/size_before

        #Save to array
        accuracies.append(accurracy)
        size_reductions.append(size_reduction)

    print(f"End set {test_set}, calculate mean")

    if len(accuracies) != 0 and len(size_reductions) !=0:
        mean_accuracy = np.mean(accuracies)
        mean_size_reduction = np.mean(size_reductions)

        calculated_sets.append(test_set)
        mean_accuracies.append(mean_accuracy)
        mean_size_reductions.append(mean_size_reduction)

result_dataframe = pd.DataFrame({"set_name": calculated_sets, "Accuracies": mean_accuracies, "Size reduction": mean_size_reductions})
result_dataframe.to_csv(os.path.join(result_path, f"standard_{settings.algorithm_type}_results.csv"), index=False)