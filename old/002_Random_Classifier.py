
import os
import warnings
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier

from config.appsettings import Appsettings
from read_files import ImportFileToDatasetWithCommas, PrepareData
from type_dictionaries import ProcessStep, RunMode

settings = Appsettings(RunMode.Random, ProcessStep.Random)
warnings.filterwarnings("ignore")



#---Debug only
test_sets = settings.sets
except_for = []
result_list = [item for item in test_sets if item not in except_for]
for test_set in result_list:

    print(f"Start for {test_set}")

    mean_accuracies = []
    paths = settings.get_paths(test_set)

    for i in range(1, settings.folds+1):

        accuracies = []
        size_reductions = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] #[0]

        train_path = os.path.join(settings.original_directory, test_set, f"{settings.folds}_tra", f"{test_set}-{settings.folds}-{i}tra.csv")

        if not os.path.exists(train_path):
            print(f"No train set for: {test_set}, {settings.folds}, iteration: {i}")
            continue


        train_set = ImportFileToDatasetWithCommas(train_path)
        train_set.index = train_set.index - 1

        for index, size in enumerate(size_reductions):

            knn_classifier = KNeighborsClassifier(n_neighbors=1)

            print(f"i: {i}, size_reduction: {size}")

            percent_train_set = train_set.sample(frac=(1-size))

            x_train = percent_train_set.iloc[:, :-1] 
            y_train = percent_train_set.iloc[:, -1:]

            knn_classifier.fit(x_train, y_train)

            test_path = os.path.join(paths["fold_test_set"], f"{test_set}-{settings.folds}-{i}tst.csv")
            
            test_set_load = ImportFileToDatasetWithCommas(test_path)
            test_set_load.index = test_set_load.index - 1

            x_test, y_test = PrepareData(test_set_load)

            prediction = knn_classifier.predict(x_test.values)
            accurracy = sklearn.metrics.accuracy_score(y_test, prediction)
            accuracies.append(accurracy)



            if len(mean_accuracies) < len(size_reductions):
                mean_accuracies.append(accurracy)
            else:
                mean_accuracies[index] = mean_accuracies[index] + accurracy

        result_dataframe = pd.DataFrame({"Accuracies": accuracies, "Size reduction": size_reductions})
        result_dataframe.to_csv(os.path.join(paths["destination"], f"random_{test_set}_{settings.folds}-{i}.csv"), index=False)

    mean_accuracies = [x/settings.folds for x in mean_accuracies]

    if(len(mean_accuracies) != len(size_reductions)):
        continue

    mean_dataframe = pd.DataFrame({"Accuracies": mean_accuracies, "Size reduction": size_reductions})
    mean_dataframe.to_csv(os.path.join(paths["destination"], f"mean_random_{test_set}_{settings.folds}.csv"), index=False)
    
    
print(f"End {test_set}")
