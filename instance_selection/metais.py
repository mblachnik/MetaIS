import pickle
import statistics
from typing import List, Union
import pandas as pd
import numpy as np
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.utils._param_validation import HasMethods
from imblearn.under_sampling.base import BaseUnderSampler
from sklearn.neighbors import NearestNeighbors
from instance_selection.meta_attributes_enum import MetaAttributesEnum
import warnings


def normalizeDistanceAttributes(df, distanceAttributes=("meanDistanceAnyClass",
                                                        "meanDistanceOppositeClass",
                                                        "meanDistanceSameClass",
                                                        "minDistanceSameClass",
                                                        "minDistanceOppositeClass",
                                                        "minDistanceAnyClass")):
    """
    The function is used to normalize the distribution of a single attribute. It subtracts median and divides by IQR.
    Here we use median and IQR because distance based attributes usually have squed distribution, so that way it is safer
    :param df: the dataset in dataframe format
    :param distanceAttributes: elements of attribute names, this list or tuple is an element which must be contained by
    a real attribute. It is used to simplify entering feature names, espacielly that ofthen we do it for multiple k which
    is stored in feature name
    :return: normalized dataset
    """
    cols = [c for c in df.columns for n in distanceAttributes if n in c]
    for c in cols:
        X = df.loc[:, c]
        idx = X != -1
        X = X[idx]
        quantils_val = X.quantile([0.25, 0.5, 0.75])
        norm_val = quantils_val.values[2] - quantils_val.values[0]
        mean = quantils_val.values[1]
        df.loc[idx, c] = (X - mean) / norm_val
        df.loc[~idx, c] = -100  # -100 is a magic number it can be replaced by any other number it should indicate that
        # the true value is missing
    return df


class ISMetaAttributesTransformer(BaseEstimator, TransformerMixin):
    """
    Class responsible for generating metaatributes from input data.
    The meta-attributes should represent average distance to its neighbors,
    average distance to its nearest enemy or nearest neighbor to the same class
    """

    def __init__(self, k_values: list[int] = [3, 5, 9, 15, 23, 33]):
        self.metaAttributTransformers = MetaAttributesEnum.generateColumns(k_values)
        self.k_values = k_values

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if type(X) is pd.DataFrame:
            X = X.values
        data_len = len(X)
        max_k = max(self.k_values) + 1
        is_data_len_smaller_than_max_k = data_len < max_k

        n = data_len if is_data_len_smaller_than_max_k else max_k
        neigh = NearestNeighbors(n_neighbors=n, metric="sqeuclidean")
        neigh.fit(X, y)

        newX = dict()

        for metaParam in self.metaAttributTransformers:
            newX[metaParam] = []

        arguments_count = X.shape[1]
        distances, indices = neigh.kneighbors(X, return_distance=True)

        for index, row in enumerate(X):
            sameRowFound = False
            anyClassDistances = []
            sameClassDistances = []
            oppositeClassDistances = []
            sameClassNeighborsCount = 0
            firstSameClassNeighborFound = False
            firstOppositeClassNeighborFound = False
            neigh_indices = indices[index]
            neigh_distances = distances[index]

            for i in range(n):
                neigh_index = neigh_indices[i]
                if sameRowFound == False:
                    if neigh_index == index:
                        sameRowFound = True
                        continue
                    elif i == n - 1:
                        break

                is_same_class = y[neigh_index] == y[index]
                normalized_distance = neigh_distances[i] / arguments_count
                anyClassDistances.append(normalized_distance)
                if is_same_class:
                    sameClassNeighborsCount += 1
                    sameClassDistances.append(normalized_distance)
                    if firstSameClassNeighborFound == False:
                        newX[MetaAttributesEnum.minDistanceSameClass.value].append(normalized_distance)
                        if firstOppositeClassNeighborFound == False:
                            newX[MetaAttributesEnum.minDistanceAnyClass.value].append(normalized_distance)
                        firstSameClassNeighborFound = True
                else:
                    oppositeClassDistances.append(normalized_distance)
                    if firstOppositeClassNeighborFound == False:
                        newX[MetaAttributesEnum.minDistanceOppositeClass.value].append(normalized_distance)
                        if firstSameClassNeighborFound == False:
                            newX[MetaAttributesEnum.minDistanceAnyClass.value].append(normalized_distance)
                        firstOppositeClassNeighborFound = True

                current_k = i if sameRowFound else i + 1
                if current_k in self.k_values:
                    strK = str(current_k)
                    newX[MetaAttributesEnum.sameClassNeighbors(strK)].append(sameClassNeighborsCount)
                    newX[MetaAttributesEnum.oppositeClassNeighbors(strK)].append(current_k - sameClassNeighborsCount)
                    newX[MetaAttributesEnum.meanDistanceAnyClass(strK)].append(statistics.mean(anyClassDistances))
                    newX[MetaAttributesEnum.meanDistanceOppositeClass(strK)].append(
                        statistics.mean(oppositeClassDistances) if firstOppositeClassNeighborFound else -1)
                    newX[MetaAttributesEnum.meanDistanceSameClass(strK)].append(
                        statistics.mean(sameClassDistances) if firstSameClassNeighborFound else -1)

            if firstSameClassNeighborFound == False:
                newX[MetaAttributesEnum.minDistanceSameClass.value].append(-1)
            elif firstOppositeClassNeighborFound == False:
                newX[MetaAttributesEnum.minDistanceOppositeClass.value].append(-1)

        if (is_data_len_smaller_than_max_k):
            for metaParam in self.metaAttributTransformers:
                if (len(newX[metaParam]) == 0):
                    for i in range(n):
                        newX[metaParam].append(-1)

        result = pd.DataFrame.from_dict(newX)
        result = normalizeDistanceAttributes(result)
        return result


class MetaIS(BaseUnderSampler):
    _parameter_constraints: dict = {
        **BaseUnderSampler._parameter_constraints,
        "estimator": [HasMethods(["fit", "predict"]), None],
        "random_state": ["random_state"],
    }

    def __init__(
            self,
            *,
            sampling_strategy="auto",
            estimator_src: Union[str, List[str]] = ["models/model.pickl"],
            threshold: float = 0.7,
            keep_proba: bool = False,
            meta_attributes_transformer=ISMetaAttributesTransformer()
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.threshold = threshold
        self.estimator_src: list[str] = [estimator_src] if isinstance(estimator_src, str) else estimator_src
        self.meta_attributes_transformer = meta_attributes_transformer
        self._estimators = None
        self.keep_proba = keep_proba
        self._y_proba_ = None

    def _load_estimator(self):
        """
        Private function which loads the model from pickle file
        If the model is saved in a different way (not using pickle) you need to update,
        overload this function
        :return:
        """
        models = []
        for src in self.estimator_src:
            with open(src, 'rb') as f:
                models.append(pickle.load(f))
        return models

    def __resample(self, X, y, yp):
        ys = yp[:, 1] > self.threshold
        if not np.any(ys):  # In case the threshold removes all samples keep at least one sample
            ys[np.argmax(yp[:, 0])] = True  # The one with highest prob for negative class is keeped
        if type(X) is pd.DataFrame:
            X = X.values
        X_resampled = np.array(X[ys, :], dtype=X.dtype)
        y_resampled = np.array(y[ys], dtype=y.dtype)
        return X_resampled, y_resampled

    def _fit_resample(self, X, y):
        """
        Function which performs instace selection
        First generate metaatributes, which are then used as an input to the instance selection classifier.
        The classifier returns probability which sayes how likely given instance is important for the training processs
        Finally according to the probability, all sample which has probability above the threshold are used to constitute
         the final training set using classical input attributes taken form X
        :param X:
        :param y:
        :return:
        """
        if self._estimators is None:
            self._estimators = self._load_estimator()
        X_meta = self.meta_attributes_transformer.transform(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yps = []
            for estimator in self._estimators:
                temp = estimator.predict_proba(X_meta)
                yps.append(temp)
            
            yps_np = np.array(yps)
            yp = np.mean(yps_np, axis=0)
            if self.keep_proba:
                self._y_proba_ = yp

        return self.__resample(X, y, yp)

    def resample_with_new_threshold(self, X, y, threshold:float):
        if (not self.keep_proba) or (self._y_proba_ is None):
            raise Exception("First you need to execute fit_resample with keep_proba set to true")
        self.threshold = threshold
        return self.__resample(X, y, self._y_proba_)
