from sklearn.ensemble import RandomForestClassifier
from research.basics.main_apply_metaIS import applyMetaIS

applyMetaIS(RandomForestClassifier(n_estimators=100, random_state=42))