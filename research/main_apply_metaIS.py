import sklearn.neighbors as knn
from research.basics.main_apply_metaIS import applyMetaIS

applyMetaIS(knn.KNeighborsClassifier(n_neighbors=1))