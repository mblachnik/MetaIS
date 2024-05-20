import os
import warnings

from matplotlib import pyplot as plt
import pandas as pd
from config.appsettings import Appsettings
from read_files import ImportFileToDatasetWithCommas
from type_dictionaries import ProcessStep, RunMode
import seaborn as sns

settings = Appsettings(RunMode.TestRun, ProcessStep.TestRun)
warnings.filterwarnings("ignore")

def generate_graph(mean_dataframe : pd.DataFrame, random_dataframe : pd.DataFrame, random_0_df: pd.DataFrame, set_name, paths):

    sns.lineplot(x='Size reduction', y='Accuracies', data=mean_dataframe, marker='o', linestyle='-', label='Meta-model', color = 'blue')
    sns.lineplot(x='Size reduction', y='Accuracies', data=random_dataframe, label='Random', marker = 'o', linestyle='-', color = 'green')
    
    plt.xlabel('Size reduction')
    plt.ylabel('Accuracies')
    plt.title(f'{set_name} - {settings.algorithm_type}')

    graph_path = os.path.join(f"Filtered by {settings.algorithm_type}", "pictures_moretheta", f"{set_name}.png")
    point_path = os.path.join(settings.weights_directory, "Standard_Results", f"standard_{settings.algorithm_type}_results.csv")
    point_set = ImportFileToDatasetWithCommas(point_path, index = False)

    point = point_set[point_set['set_name'] == set_name].squeeze()
    
    plt.scatter(point['Size reduction'], point['Accuracies'], color='red', marker='X', s=100, label=f'{settings.algorithm_type} Result')

    line_0 = random_0_df.squeeze()
    x = line_0['Accuracies']

    plt.axhline(y=line_0['Accuracies'], color='black', linestyle='--', label='1NN')

    plt.legend()

    plt.savefig(graph_path)
    plt.clf()


if not os.path.exists(os.path.join(f"Filtered by {settings.algorithm_type}", "pictures_moretheta")):
    os.makedirs(os.path.join(f"Filtered by {settings.algorithm_type}", "pictures_moretheta"))

#test_sets = ["banana", "australian", "phoneme", "titanic"]
test_sets = settings.sets
except_for = []
result_list = [item for item in test_sets if item not in except_for]
for test_set in result_list:
    paths = settings.get_paths(test_set)

    mean_path = os.path.join(paths["destination"], f"mean_{test_set}_{settings.folds}.csv")
    random_path = os.path.join(settings.weights_directory, test_set, "random", f"mean_random_{test_set}_{settings.folds}.csv")
    random_0_path = os.path.join(settings.weights_directory, test_set, "random", f"0_mean_random_{test_set}_{settings.folds}.csv")

    if not os.path.exists(mean_path):
        print(f"Current test set: {test_set} - doesn't have mean values for size reduction and accuracy calculated. Do 5_1 first.")
        continue

    if not os.path.exists(random_path):
        print(f"Current test set: {test_set} - doesn't have random results for different size reductions")
        continue
    if not os.path.exists(random_0_path):
        print(f"Current test set: {test_set} - doesn't have random results 0 size reductions")
        continue

    mean_dataframe = ImportFileToDatasetWithCommas(mean_path)
    mean_random = ImportFileToDatasetWithCommas(random_path)
    random_0 = ImportFileToDatasetWithCommas(random_0_path, index=False)

    generate_graph(mean_dataframe, mean_random, random_0, test_set, paths)