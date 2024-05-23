import pickle

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

VOTING_KIND = ("auto", "hard", "soft")

class ISMetaAttributesTransformer(BaseEstimator, TransformerMixin):
    """
    Class responsible for generating metaatributes from input data.
    The meta-attributes should represent average distance to its neighbors,
    average distance to its nearest enemy or nearest neighbor to the same class
    """
    def __init__(self, by=1, columns=None):
        self.by = by
        self.columns = columns
        self.metaAttributTransformers = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #The implementation comes here
        return X


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





