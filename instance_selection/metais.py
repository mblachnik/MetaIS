import pickle
import statistics

import pandas as pd
import numpy as np
import numpy as np

from scipy import sparse
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring
from imblearn.utils._param_validation import HasMethods, StrOptions
from imblearn.under_sampling.base import BaseUnderSampler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

from instance_selection.meta_attributes_enum import MetaAttributesEnum



VOTING_KIND = ("auto", "hard", "soft")

class ISMetaAttributesTransformer(BaseEstimator, TransformerMixin):
    """
    Class responsible for generating metaatributes from input data.
    The meta-attributes should represent average distance to its neighbors,
    average distance to its nearest enemy or nearest neighbor to the same class
    """
    def __init__(self, by=1, columns=None, k_values=[3,5,9,15,23,33]):
        self.by = by
        self.columns = columns
        self.metaAttributTransformers = [
            #MetaAttributesEnum.id.value,
            MetaAttributesEnum.minDistanceSameClass.value, MetaAttributesEnum.minDistanceOppositeClass.value, MetaAttributesEnum.minDistanceAnyClass.value]
        self.k_values = k_values
        for mat in self.k_values:
            strMat = str(mat)
            self.metaAttributTransformers.append(MetaAttributesEnum.sameClassNeighbors(strMat))
            self.metaAttributTransformers.append(MetaAttributesEnum.oppositeClassNeighbors(strMat))
            self.metaAttributTransformers.append(MetaAttributesEnum.meanDistanceAnyClass(strMat))
            self.metaAttributTransformers.append(MetaAttributesEnum.meanDistanceSameClass(strMat))
            self.metaAttributTransformers.append(MetaAttributesEnum.meanDistanceOppositeClass(strMat))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, ids=None):
        if type(X) is pd.DataFrame:
            X = X.values
        data_len = len(X)
        max_k = max(self.k_values) + 1
        is_data_len_smaller_than_max_k = data_len < max_k

        n = data_len if is_data_len_smaller_than_max_k else max_k
        neigh = NearestNeighbors(n_neighbors=n, metric="sqeuclidean") 
        neigh.fit(X,y)

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
            #newX[MetaAttributesEnum.id.value].append(ids[index])
            neigh_indices = indices[index]
            neigh_distances = distances[index]

            for i in range(n):
                id = neigh_indices[i]
                if sameRowFound == False:
                    if id == index:
                        sameRowFound = True
                        continue
                    elif i == n - 1:
                        break

                is_same_class = y[id] == y[index]
                normalized_distance = neigh_distances[i]/arguments_count
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
                    newX[MetaAttributesEnum.meanDistanceOppositeClass(strK)].append(statistics.mean(oppositeClassDistances) if firstOppositeClassNeighborFound else -1)
                    newX[MetaAttributesEnum.meanDistanceSameClass(strK)].append(statistics.mean(sameClassDistances) if firstSameClassNeighborFound else -1)

            if firstSameClassNeighborFound == False:
                newX[MetaAttributesEnum.minDistanceSameClass.value].append(-1)
            elif firstOppositeClassNeighborFound == False:
                newX[MetaAttributesEnum.minDistanceOppositeClass.value].append(-1)

        if(is_data_len_smaller_than_max_k):
            for metaParam in self.metaAttributTransformers:
                if(len(newX[metaParam]) == 0):
                    for i in range(n):
                        newX[metaParam].append(-1)
        
        result = pd.DataFrame.from_dict(newX)
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
            estimator_src: str = "models/model.pickl",
            threshold: float =0.7,
            meta_attributes_transformer=ISMetaAttributesTransformer()
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.threshold = threshold
        self.estimator_src: str = estimator_src
        self.meta_attributes_transformer = meta_attributes_transformer
        self._estimator = None

    def _load_estimator(self):
        """
        Private function which loads the model from pickle file
        If the model is saved in a different way (not using pickle) you need to update,
        overload this function
        :return:
        """
        with open(self.estimator_src, 'rb') as f:
             model = pickle.load(f)
             self._estimator = model
        return model

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
        if self._estimator is None:
            self._estimator = self._load_estimator()
        X_meta = self.meta_attributes_transformer.transform(X, y)
        yp = self._estimator.predict_proba(X_meta)
        ys = yp[:, 0] > self.threshold
        X_resampled = np.array(X[ys, :], dtype=X.dtype)
        y_resampled = np.array(y[ys], dtype=y.dtype)
        return X_resampled, y_resampled





