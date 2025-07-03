from sklearn.svm import SVC
from research.basics.main_apply_metaIS import applyMetaIS

applyMetaIS(SVC(kernel='rbf', C=1.0, gamma='scale'))