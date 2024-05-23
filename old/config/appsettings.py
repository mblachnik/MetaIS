import os
from type_dictionaries import ProcessStep, RunMode


def init_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

class Appsettings:
    algorithm_type = "ICF"
    folds = 5
    weights_directory = f"Filtered by {algorithm_type}"
    original_directory = "Original Datasets"

    meta_suffix = "_meta.csv"
    meta_weights_suffix = "_meta_weights.csv"

    #Possible folder names which are not set name

    _workspaces = ["metadata"]

    # Result folders

    _fullset_metadata = os.path.join(original_directory, "metadata", "fullset")
    _fullset_meta_weights = os.path.join(
        weights_directory, "meta_weights", "fullset_meta_weights"
    )

    _fold_metadata = os.path.join(original_directory, "metadata", f"{folds}_tra")
    _fold_meta_weights = os.path.join(
        weights_directory, "meta_weights", f"{folds}_tra_meta_weights"
    ) 

    _random_forest_models = os.path.join(weights_directory, "RF_models")

    #Constructor

    def __init__(self, run_mode: RunMode, step: ProcessStep):
        self.run_mode = run_mode
        self.step = step

        print(f"App works in mode: {run_mode.name}. Process step: {step.name}")

        self.sets = [f for f in os.listdir(self.original_directory) if (not f[0] == '.' and f not in [self._workspaces])]
        
    def get_paths(self, folder):
        result: dict

        if self.step.value == 1:
            result = self.__generate_metadata_paths(folder)
        elif self.step.value == 2:
            result = self.__combine_proto_with_meta_paths(folder)
        elif self.step.value == 3:
            result = self.__test_model_paths(folder)
        elif self.step.value == 4:
            result = self.__random_results(folder)

        if  "destination" in result and result["destination"] != "":
            init_folder(result["destination"])

        return result

    #Paths for different processes

    def __generate_metadata_paths(self, set_folder):
        if self.run_mode.value == 1:
            source = os.path.join(self.original_directory, set_folder, "fullset")
            destination = self._fullset_metadata

        elif self.run_mode.value == 2:
            source = os.path.join(
                self.original_directory, set_folder, f"{self.folds}_tra"
            )
            destination = os.path.join(self._fold_metadata, set_folder)

        return {"source": source, "destination": destination}

    def __combine_proto_with_meta_paths(self, set_folder):
        if self.run_mode.value == 1:
            source_meta = os.path.join(self.original_directory, "metadata", "fullset")
            source_weights = os.path.join(
                self.weights_directory, set_folder, "fullset_proto"
            )
            destination = self._fullset_meta_weights

        elif self.run_mode.value == 2:
            source_meta = os.path.join(
                self.original_directory, "metadata", f"{self.folds}_tra", set_folder
            )
            source_weights = os.path.join(
                self.weights_directory, set_folder, f"{self.folds}_tra_proto"
            )
            destination = os.path.join(self._fold_meta_weights, set_folder)

        return {
            "source_meta": source_meta,
            "source_weights": source_weights,
            "destination": destination,
        }

    def __test_model_paths(self, test_folder):
        fold_test_set = os.path.join(
            self.original_directory, test_folder, f"{self.folds}_tst"
        )
        destination = os.path.join(self.weights_directory, test_folder, "results", "morethera")

        init_folder(self._random_forest_models)

        return {
            "fullset_metadata": self._fullset_metadata,
            "fullset_meta_weights": self._fullset_meta_weights,
            "fold_metadata": self._fold_metadata,
            "fold_meta_weights": self._fold_meta_weights,
            "fold_test_set": fold_test_set,
            "random_forest_models": self._random_forest_models,
            "destination": destination
        }
    
    def __random_results(self, test_folder):
        fold_test_set = os.path.join(
            self.original_directory, test_folder, f"{self.folds}_tst"
        )
        destination = os.path.join(self.weights_directory, test_folder, "random")

        return {
            "fold_test_set": fold_test_set,
            "destination": destination
        }